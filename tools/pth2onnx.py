from __future__ import absolute_import

import sys
sys.path.append('.')
import os
import torch
import torch.nn as nn
# from siamfc.backbones import AlexNetV1_Test   # for pruning
from siamfc.backbones import AlexNetV1
from siamfc.heads import SiamFC


class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


if __name__ == '__main__':
    # setup GPU device
    device = torch.device('cuda:0')

    pth_path = 'pretrained/siamfc_alexnet_e50.pth'
    if os.path.exists(pth_path):
        print('Convert {} to onnx model'.format(pth_path))
        # load pth model
        net = Net(
            # backbone=AlexNetV1_Test(),    # for pruning
            backbone=AlexNetV1(),
            head=SiamFC(0.001))

        net.load_state_dict(torch.load(pth_path, map_location=lambda storage, loc: storage))
        net = net.to(device)

        # convert model
        # dynamic input
        onnx_path = 'pretrained/siamfc_alexnet_pruning_e50_dynamic.onnx'
        if not os.path.exists(onnx_path):
            dummy_input = torch.randn(1, 3, 127, 127).to(device)
            input_names = ['input']
            output_names = ['output']
            dynamic_axes = {'input':{0:'batch_size', 2:'width', 3:'height'}, 'output':{0:'batch_size', 2:'width', 3:'height'}} #adding names for better debugging
            torch.onnx.export(net.backbone, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)
            print('dynamic input done.')

        # z input
        onnx_path = 'pretrained/siamfc_alexnet_pruning_e50_z.onnx'
        if not os.path.exists(onnx_path):
            dummy_input = torch.randn(1, 3, 127, 127).to(device)
            input_names = ['input']
            output_names = ['output']
            torch.onnx.export(net.backbone, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
            print('z input done.')

        # x input
        onnx_path = 'pretrained/siamfc_alexnet_pruning_e50_x.onnx'
        if not os.path.exists(onnx_path):
            dummy_input = torch.randn(3, 3, 255, 255).to(device)
            input_names = ['input']
            output_names = ['output']
            torch.onnx.export(net.backbone, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
            print('x input done.')

    else:
        print("File {} dose not exist".format(pth_path))
