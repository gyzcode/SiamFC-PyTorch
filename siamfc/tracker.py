from __future__ import absolute_import

import torch
import os
import cv2
import numpy as np
import tensorrt as trt
from collections import namedtuple
from got10k.trackers import Tracker
from . import ops
from .heads import SiamFC
import pycuda.autoinit
import pycuda.driver as cuda


__all__ = ['TrackerSiamFC1']

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def get_engine(engine_file_path=""):
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
            # print('done.')
            # return engine
    else:
        print("File {} dose not exist".format(engine_file_path))


def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    for binding in engine:
        if engine.binding_is_input(binding):
            shape = context.get_binding_shape(0)
        else:
            shape = context.get_binding_shape(1)
            shape_of_output = shape
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, shape_of_output


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs   


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrackerSiamFC1(Tracker):

    def __init__(self, engine_path=None, **kwargs):
        super(TrackerSiamFC1, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)
        
        # setup GPU device
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            print('cuda is not available, exit.')
            return
        self.device = torch.device('cuda:0')
     
        # setup model
        self.head = SiamFC(self.cfg.out_scale)
        self.head = self.head.to(self.device)

        # deal with Cudnn Error in configure: 7 (CUDNN_STATUS_MAPPING_ERROR)
        dummy_input = torch.randn(1, 1, 1, 1).to(self.device)
        self.head(dummy_input, dummy_input)

        self.engine = get_engine(engine_path)

        # Create the context for this engine
        self.stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.profile_shapes = self.engine.get_profile_shape(0, 0)

        self.tracking = False

    
    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)


    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.head.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] + box[3] / 2,
            box[0] + box[2] / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # Allocate buffers for input and output
        self.context.set_binding_shape(0, self.profile_shapes[0])
        self.inputs, self.outputs, self.bindings, self.shape_of_output = allocate_buffers(self.engine, self.context) # input, output: host # bindings

        # Tensorrt inferrence
        # Load data to the buffer
        self.inputs[0].host = np.expand_dims(np.transpose(z, [2, 0, 1]), axis=0).astype(np.float32).reshape(-1)
        # Do inference
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream) # numpy data
        feat = postprocess_the_outputs(trt_outputs[0], self.shape_of_output)
        self.kernel = torch.from_numpy(feat).to(self.device)

        # Allocate buffers for input and output
        self.context.set_binding_shape(0, self.profile_shapes[1])
        self.inputs, self.outputs, self.bindings, self.shape_of_output = allocate_buffers(self.engine, self.context) # input, output: host # bindings

    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.head.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)

        # Tensorrt inferrence
        # Load data to the buffer
        self.inputs[0].host = np.transpose(x, [0, 3, 1, 2]).astype(np.float32).reshape(-1)
        # Do inference
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream) # numpy data
        feat = postprocess_the_outputs(trt_outputs[0], self.shape_of_output)
        x = torch.from_numpy(feat).to(self.device)

        responses = self.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
