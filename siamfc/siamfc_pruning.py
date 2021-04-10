from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
#from apex import amp

__all__ = ['TrackerSiamFC']

################## 共有三处需要改动 #######################

### 第1处. ########### 定义梯度反向传播时的钩子函数，用来将下次要裁剪的权重梯度清零（裁剪训练时pytorch版本应该为1.3.0以下，高版本pytorch的钩子函数使用方式有所不同） ############
def hook_layers(net, ratio=0.8):

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_weight[:, ch2:, :, :] = 0.0
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[2])
        grad_weight[:, ch2:] = 0.0
        grad_bias = torch.zeros_like(grad_input[0])
        grad_bias[0:ch] = grad_input[0][0:ch]
    
        return (grad_bias, grad_input[1], grad_weight)
        
    layer0 = net.backbone.conv1[0]
    layer0.register_backward_hook(hook_f0)
    
    layer1 = net.backbone.conv2[0]
    layer1.register_backward_hook(hook_f1)

    layer2 = net.backbone.conv3[0]
    layer2.register_backward_hook(hook_f1)

    layer3 = net.backbone.conv4[0]
    layer3.register_backward_hook(hook_f1)

    #layer4 = net.backbone.conv5[0]
    #layer4.register_backward_hook(hook_f1)

class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        #### 第2处. ############ 0.64为0.8*0.8，因为一共进行两次裁剪，每次保留80%， 1.0为非衰减权重的比率，测试时不需衰减，所以为1.0 ####################
        self.net = Net(
            backbone=AlexNetV1(.64, 1.0),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # convert to caffe model
        # sm = torch.jit.script(self.net)
        # sm.save("siamfc_model.pt")

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # apex initialization
        # opt_level = 'O1'
        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

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
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
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
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)

        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
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
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            #### 第3处. #################################### 训练时进行裁剪 ########################################
            if epoch in prun_epoch:
                rate *= 0.8
                if epoch == max(prun_epoch):
                    t_net = AlexNetV1(rate, 1.0)
                else:
                    t_net = AlexNetV1(rate, 0.8)
                t_net_dict = t_net.state_dict()

                print(t_net)
                cout, cin, ch, cw = -1, -1, -1, -1
                for name1, item1 in self.net.backbone.named_parameters():
                    for name2, item2 in t_net.named_parameters():
                        if name1 == name2:
                            if item1.dim() > 2:
                                if cout > 0:
                                    t_net_dict[name2] = item1[0:item2.size(0), 0:cout, :, :]
                                else:
                                    t_net_dict[name2] = item1[0:item2.size(0), :, :, :]
                                cout, cin, ch, cw = item2.size()

                            elif item1.dim() > 1 and item1.dim() < 3:
                                t_net_dict[name2] = item1[:, 0:item2.size(1)]
                            else:
                                if 'fc' in name2:
                                    t_net_dict[name2] = item1[:]
                                else:
                                    t_net_dict[name2] = item1[0:item2.size(0)]      

                t_net.load_state_dict(t_net_dict)
                self.net = Net(
                    backbone=t_net,
                    head=SiamFC(self.cfg.out_scale))

                self.optimizer = optim.SGD(
                    self.net.parameters(),
                    lr=self.cfg.initial_lr,
                    weight_decay=self.cfg.weight_decay,
                    momentum=self.cfg.momentum)

                gamma = np.power(
                    self.cfg.ultimate_lr / self.cfg.initial_lr,
                    1.0 / self.cfg.epoch_num)
                self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

                self.lr_scheduler.step(epoch=epoch)
            
                device = 'cuda' 
                # data parallel for multiple-GPU
                if device == 'cuda':
                    cudnn.benchmark = True
                self.net.to(self.device)

                self.last_ep = epoch
            
            if epoch < max(prun_epoch):
                hook_layers(self.net, 0.8)


            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels