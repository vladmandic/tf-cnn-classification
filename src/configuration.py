
# Training Parameters
DEVICE = "gpu" # cpu or gpu
EPOCHS = 10 # how many epochs to run while training
BATCH_SIZE = 8 # how many images to train on in a single call
NUM_CLASSES = 4
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3 # rgb
TRAIN_SET_RATIO = 0.6 # 60% of dataset for training
TEST_SET_RATIO = 0.2 # 20% of dataset for testing
SAVE_N_EPOCH = 5 # save checkpoint every n epochs

# Model Network
# 0: MobileNet-v1, 1: MobileNet-v2, 2: MobileNet-v3-Large, 3: MobileNet-v3-Small
# 4: EfficientNet-B0, 5: EfficientNet-B1, 6: EfficientNet-B2, 7: EfficientNet-B3
# 8: EfficientNet-B4, 9: EfficientNet-B5, 10: EfficientNet-B6, 11: EfficientNet-B7
# 12: ResNeXt50, 13: ResNeXt101
# 14: InceptionV4, 15: InceptionResNetV1, 16: InceptionResNetV2
# 17: SE_ResNet_50, 18: SE_ResNet_101, 19: SE_ResNet_152
# 20: SqueezeNet
# 21: DenseNet_121, 22: DenseNet_169, 23: DenseNet_201, 24: DenseNet_269
# 25: ShuffleNetV2-0.5x, 26: ShuffleNetV2-1.0x, 27: ShuffleNetV2-1.5x, 28: ShuffleNetV2-2.0x
# 29: ResNet_18, 30: ResNet_34, 31: ResNet_50, 32: ResNet_101, 33: ResNet_152
# 34: SEResNeXt_50, 35: SEResNeXt_101
# 36: RegNet
MODEL = 3

# Locations
save_model_dir = "saved/"
test_image_dir = ""
dataset_dir = "train/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
