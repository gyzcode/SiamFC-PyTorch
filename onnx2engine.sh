#!/bin/sh
# trtexec --onnx=pretrained/siamfc_alexnet_pruning_e50_dynamic.onnx --explicitBatch --minShapes='input':1x3x127x127 --optShapes='input':3x3x255x255 --maxShapes='input':3x3x255x255 --shapes='input':3x3x255x255 --workspace=1024 --fp16 --int8 --calib=pretrained/CalibrationTableSiamfc_pruning --saveEngine=pretrained/siamfc_alexnet_pruning_e50_dynamic_int8.engine

# trtexec --onnx=pretrained/siamfc_alexnet_pruning_e50_z.onnx --workspace=1024 --fp16 --int8 --calib=pretrained/CalibrationTableSiamfc_pruning --saveEngine=pretrained/siamfc_alexnet_pruning_e50_z_int8.engine

trtexec --onnx=pretrained/siamfc_alexnet_pruning_e50_x.onnx --maxBatch=3 --workspace=1024 --fp16 --int8 --calib=pretrained/CalibrationTableSiamfc_pruning --saveEngine=pretrained/siamfc_alexnet_pruning_e50_x_int8.engine
