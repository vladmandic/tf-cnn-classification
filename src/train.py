from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, save_model_dir, MODEL, SAVE_N_EPOCH
from prepare_data import generate_datasets, load_and_preprocess_image
import math
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, se_resnet, squeezenet, densenet, shufflenet_v2, resnet, se_resnext
from models.RegNet import regnet
from tensorflow.python.client import device_lib 
import os

def get_model():
    if MODEL == 0:
        return mobilenet_v1.MobileNetV1()
    elif MODEL == 1:
        return mobilenet_v2.MobileNetV2()
    elif MODEL == 2:
        return mobilenet_v3_large.MobileNetV3Large()
    elif MODEL == 3:
        return mobilenet_v3_small.MobileNetV3Small()
    elif MODEL == 4:
        return efficientnet.efficient_net_b0()
    elif MODEL == 5:
        return efficientnet.efficient_net_b1()
    elif MODEL == 6:
        return efficientnet.efficient_net_b2()
    elif MODEL == 7:
        return efficientnet.efficient_net_b3()
    elif MODEL == 8:
        return efficientnet.efficient_net_b4()
    elif MODEL == 9:
        return efficientnet.efficient_net_b5()
    elif MODEL == 10:
        return efficientnet.efficient_net_b6()
    elif MODEL == 11:
        return efficientnet.efficient_net_b7()
    elif MODEL == 12:
        return resnext.ResNeXt50()
    elif MODEL == 13:
        return resnext.ResNeXt101()
    elif MODEL == 14:
        return inception_v4.InceptionV4()
    elif MODEL == 15:
        return inception_resnet_v1.InceptionResNetV1()
    elif MODEL == 16:
        return inception_resnet_v2.InceptionResNetV2()
    elif MODEL == 17:
        return se_resnet.se_resnet_50()
    elif MODEL == 18:
        return se_resnet.se_resnet_101()
    elif MODEL == 19:
        return se_resnet.se_resnet_152()
    elif MODEL == 20:
        return squeezenet.SqueezeNet()
    elif MODEL == 21:
        return densenet.densenet_121()
    elif MODEL == 22:
        return densenet.densenet_169()
    elif MODEL == 23:
        return densenet.densenet_201()
    elif MODEL == 24:
        return densenet.densenet_264()
    elif MODEL == 25:
        return shufflenet_v2.shufflenet_0_5x()
    elif MODEL == 26:
        return shufflenet_v2.shufflenet_1_0x()
    elif MODEL == 27:
        return shufflenet_v2.shufflenet_1_5x()
    elif MODEL == 28:
        return shufflenet_v2.shufflenet_2_0x()
    elif MODEL == 29:
        return resnet.resnet_18()
    elif MODEL == 30:
        return resnet.resnet_34()
    elif MODEL == 31:
        return resnet.resnet_50()
    elif MODEL == 32:
        return resnet.resnet_101()
    elif MODEL == 33:
        return resnet.resnet_152()
    elif MODEL == 34:
        return se_resnext.SEResNeXt50()
    elif MODEL == 35:
        return se_resnext.SEResNeXt101()
    elif MODEL == 36:
        return regnet.RegNet()
    else:
        raise ValueError("The MODEL does not exist.")


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


if __name__ == '__main__':
    # Disable CUDA warnings and print CUDA capable devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(device_lib.list_local_devices())

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    model = get_model()
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features, data_augmentation=True)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))

        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if epoch % SAVE_N_EPOCH == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
