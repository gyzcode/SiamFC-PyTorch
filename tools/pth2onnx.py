from __future__ import absolute_import

import sys
sys.path.append('.')
import os
import torch
import torch.nn as nn
from siamfc.backbones import AlexNetV1_Test
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

    pth_path = 'pretrained/siamfc_alexnet_pruning_e50.pth'
    if os.path.exists(pth_path):
        print('Convert {} to onnx model'.format(pth_path))
        # load pth model
        net = Net(
            backbone=AlexNetV1_Test(),
            head=SiamFC(0.001))

        net.load_state_dict(torch.load(pth_path, map_location=lambda storage, loc: storage))
        net = net.to(device)

        # convert model
        onnx_path = 'pretrained/siamfc_alexnet_pruning_e50_dynamic.onnx'
        batch = 1
        width = 127
        height = 127
        dummy_input = torch.randn(batch, 3, width, height).to(device)
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input':{0:'batch_size', 2:'width', 3:'height'}, 'output':{0:'batch_size', 2:'width', 3:'height'}} #adding names for better debugging
        torch.onnx.export(net.backbone, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)
        print('done.')
    else:
        print("File {} dose not exist".format(pth_path))
