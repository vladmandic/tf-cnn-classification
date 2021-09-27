# Classification CNNs in TensorFlow

## Configuration

- Edit model image sizes, model type, classes, epochs, etc:  
  `configuration.py`

### Supported Networks

> 0: MobileNet-v1, 1: MobileNet-v2, 2: MobileNet-v3-Large, 3: MobileNet-v3-Small  
4: EfficientNet-B0, 5: EfficientNet-B1, 6: EfficientNet-B2, 7: EfficientNet-B3  
8: EfficientNet-B4, 9: EfficientNet-B5, 10: EfficientNet-B6, 11: EfficientNet-B7  
12: ResNeXt50, 13: ResNeXt101  
14: InceptionV4, 15: InceptionResNetV1, 16: InceptionResNetV2  
17: SE_ResNet_50, 18: SE_ResNet_101, 19: SE_ResNet_152  
20: SqueezeNet  
21: DenseNet_121, 22: DenseNet_169, 23: DenseNet_201, 24: DenseNet_269  
25: ShuffleNetV2-0.5x, 26: ShuffleNetV2-1.0x, 27: ShuffleNetV2-1.5x, 28: ShuffleNetV2-2.0x  
29: ResNet_18, 30: ResNet_34, 31: ResNet_50, 32: ResNet_101, 33: ResNet_152  
34: SEResNeXt_50, 35: SEResNeXt_101  
36: RegNet

<br>

## Training & Evaluation

### 1. Prepare Data

- For monolithic datasets:
  - Place input data into `/dataset` with subfolder name as class name
  - Automatically split dataset into  
    `train/train`, `train/valid`, `train/test`
  > python src/split.py

- For pre-split datasets:
  - Place input data into  
  `train/train`, `train/valid`, `train/test`

- Tesize images and generate binary records  
  `train/train.tfrecord`, `train/valid.tfrecord`, `train/test.tfrecord`
  > python src/prepare.py

<br>

### 2. Edit Configuration

> src/configuration.py

- `NUM_CLASSES` must match number from previous steps
- `MODEL` choose model network architecture

<br>

### 3. Train Model

> python src/train.py

- Training saves checkpoint `saved/epoch-*` every `SAVE_N_EPOCH`
- End of training saves final checkpoint as `saved/model/*`
- End of training exports model to `saved/saved_model.pb`

<br>

### 4. Evaluate Model Precision

Evaluate the model on the test dataset
> python src/evalulate.py

Test on Single Image
> python src/predict.py [filename]

<br>

## Convert to TFJS Graph Model

*Example: Quantize to F16 and save to `saved/graph`*

> tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model \
--strip_debug_ops=* --weight_shard_size_bytes=1073741824 \
--quantize_float16=* --control_flow_v2=true \
saved/ saved/graph/

<br>

Optionally check model signature

Install [TFJS-Utils](https://github.com/vladmandic/tfjs-utils/blob/main/src/signature.js)

> node signature.js saved/graph

```js
2021-09-27 18:01:01 DATA:  created on: 2021-09-27T21:54:38.648Z
2021-09-27 18:01:01 INFO:  graph model: /home/vlado/dev/tf-cnn-classification/saved/graph/model.json
2021-09-27 18:01:01 INFO:  size: { numTensors: 51, numDataBuffers: 51, numBytes: 12780728 }
2021-09-27 18:01:01 DATA:  ops used by model: {
  graph: [ 'Const', 'Placeholder', 'Shape', 'Identity', [length]: 4 ],
  convolution: [ '_FusedConv2D', 'DepthwiseConv2dNative', 'AvgPool', [length]: 3 ],
  slice_join: [ 'GatherV2', 'ConcatV2', 'Pack', [length]: 3 ],
  reduction: [ 'Prod', [length]: 1 ],
  transformation: [ 'Reshape', [length]: 1 ],
  matrices: [ 'MatMul', [length]: 1 ],
  arithmetic: [ 'BiasAdd', [length]: 1 ],
  normalization: [ 'Softmax', [length]: 1 ]
}
2021-09-27 18:01:01 DATA:  inputs: [ { name: 'input_1', dtype: 'DT_FLOAT', shape: [ -1, 224, 224, 3 ] } ]
2021-09-27 18:01:01 DATA:  outputs: [ { id: 0, name: 'output_1', dytpe: 'DT_FLOAT', shape: [ -1, 1, 1, 4 ] } ]
```

<br><hr><br>

## References

### Papers

- [MobileNet_V1](https://arxiv.org/abs/1704.04861)
- [MobileNet_V2](https://arxiv.org/abs/1801.04381)
- [MobileNet_V3](https://arxiv.org/abs/1905.02244)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [ResNeXt](https://arxiv.org/abs/1611.05431)
- [Inception_V4/Inception_ResNet_V1/Inception_ResNet_V2](https://arxiv.org/abs/1602.07261)
- [SENet](https://arxiv.org/abs/1709.01507)
- [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [RegNet](https://arxiv.org/abs/2003.13678)

### Implementations

- [MobileNet_V3](https://github.com/calmisential/MobileNetV3_TensorFlow2)
- [EfficientNet](https://github.com/calmisential/EfficientNet_TensorFlow2)
- [ResNeXt](https://github.com/calmisential/ResNeXt_TensorFlow2)
- [InceptionV4](https://github.com/calmisential/InceptionV4_TensorFlow2)
- [DenseNet](https://github.com/calmisential/DenseNet_TensorFlow2)
- [ResNet](https://github.com/calmisential/TensorFlow2.0_ResNet)
- [AlexNet and VGG](https://github.com/calmisential/TensorFlow2.0_Image_Classification)
- [InceptionV3](https://github.com/calmisential/TensorFlow2.0_InceptionV3)
- [ResNet](https://github.com/calmisential/TensorFlow2.0_ResNet)
