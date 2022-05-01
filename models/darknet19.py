# ================================================================
#
#   Editor      : Pycharm
#   File name   : darknet19
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:58
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : darknet19网络
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class darknet_Model(tf.keras.Model):
    def __init__(self):
        super(darknet_Model, self).__init__()

        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', name='conv1', use_bias=False)
        self.bn1 = layers.BatchNormalization(name='norm1')

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', name='conv2', use_bias=False)
        self.bn2 = layers.BatchNormalization(name='norm2')

        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', name='conv3', use_bias=False)
        self.bn3 = layers.BatchNormalization(name='norm3')

        self.conv4 = layers.Conv2D(64, (1, 1), padding='same', name='conv4', use_bias=False)
        self.bn4 = layers.BatchNormalization(name='norm4')

        self.conv5 = layers.Conv2D(128, (3, 3), padding='same', name='conv5', use_bias=False)
        self.bn5 = layers.BatchNormalization(name='norm5')

        self.conv6 = layers.Conv2D(256, (3, 3), padding='same', name='conv6', use_bias=False)
        self.bn6 = layers.BatchNormalization(name='norm6')

        self.conv7 = layers.Conv2D(128, (1, 1), padding='same', name='conv7', use_bias=False)
        self.bn7 = layers.BatchNormalization(name='norm7')

        self.conv8 = layers.Conv2D(256, (3, 3), padding='same', name='conv8', use_bias=False)
        self.bn8 = layers.BatchNormalization(name='norm8')

        self.conv9 = layers.Conv2D(512, (3, 3), padding='same', name='conv9', use_bias=False)
        self.bn9 = layers.BatchNormalization(name='norm9')

        self.conv10 = layers.Conv2D(256, (1, 1), padding='same', name='conv10', use_bias=False)
        self.bn10 = layers.BatchNormalization(name='norm10')

        self.conv11 = layers.Conv2D(512, (3, 3), padding='same', name='conv11', use_bias=False)
        self.bn11 = layers.BatchNormalization(name='norm11')

        self.conv12 = layers.Conv2D(256, (1, 1), padding='same', name='conv12', use_bias=False)
        self.bn12 = layers.BatchNormalization(name='norm12')

        self.conv13 = layers.Conv2D(512, (3, 3), padding='same', name='conv13', use_bias=False)
        self.bn13 = layers.BatchNormalization(name='norm13')

        self.conv14 = layers.Conv2D(1024, (3, 3), padding='same', name='conv14', use_bias=False)
        self.bn14 = layers.BatchNormalization(name='norm14')

        self.conv15 = layers.Conv2D(512, (1, 1), padding='same', name='conv15', use_bias=False)
        self.bn15 = layers.BatchNormalization(name='norm15')

        self.conv16 = layers.Conv2D(1024, (3, 3), padding='same', name='conv16', use_bias=False)
        self.bn16 = layers.BatchNormalization(name='norm16')

        self.conv17 = layers.Conv2D(512, (1, 1), padding='same', name='conv17', use_bias=False)
        self.bn17 = layers.BatchNormalization(name='norm17')

        self.conv18 = layers.Conv2D(1024, (3, 3), padding='same', name='conv18', use_bias=False)
        self.bn18 = layers.BatchNormalization(name='norm18')

        self.conv19 = layers.Conv2D(5 * (5 + 2), (1, 1), padding='same', name='conv19')
        # self.conv19 = layers.Conv2D(5 * (5 + self.numwork), (1, 1), padding='same', name='conv22', use_bias=False)

        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2))

        self.leakrelu = layers.LeakyReLU(alpha=0.1)

    def call(self, inputs,training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # unit 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # unit 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.leakrelu(x)
        # unit 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.leakrelu(x)
        # unit 5
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # unit 6
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.leakrelu(x)
        # unit 7
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.leakrelu(x)
        # unit 8
        x = self.conv8(x)
        x = self.bn8(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # unit 9
        x = self.conv9(x)
        x = self.bn9(x, training=training)
        x = self.leakrelu(x)
        # unit 10
        x = self.conv10(x)
        x = self.bn10(x, training=training)
        x = self.leakrelu(x)
        # unit 11
        x = self.conv11(x)
        x = self.bn11(x, training=training)
        x = self.leakrelu(x)
        # unit 12
        x = self.conv12(x)
        x = self.bn12(x, training=training)
        x = self.leakrelu(x)
        # unit 13
        x = self.conv13(x)
        x = self.bn13(x, training=training)
        x = self.leakrelu(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        skip_x = x  # [b,16,16,512]

        # unit 14
        x = self.conv14(x)
        x = self.bn14(x, training=training)
        x = self.leakrelu(x)
        # unit 15
        x = self.conv15(x)
        x = self.bn15(x, training=training)
        x = self.leakrelu(x)
        # unit 16
        x = self.conv16(x)
        x = self.bn16(x, training=training)
        x = self.leakrelu(x)
        # unit 17
        x = self.conv17(x)
        x = self.bn17(x, training=training)
        x = self.leakrelu(x)
        # unit 18
        x = self.conv18(x)
        x = self.bn18(x, training=training)
        x = self.leakrelu(x)
        # scale
        x = tf.concat([x, skip_x], axis=-1)  # [b,16,16,1024+512]
        # unit 19
        x = self.conv19(x)

        out = layers.Reshape([np.shape(x)[1], np.shape(x)[2], 5, 5 + 2])(x)

        return out

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4

if __name__ == '__main__':
    x = tf.random.normal([3, 512, 512, 3])
    model = darknet_Model()
    print(np.shape(model(x)))

    weight_reader = WeightReader('../yolo.weights')

    weight_reader.reset()
    nb_conv = 19
    GRIDSZ=16

    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv' + str(i))
        conv_layer.trainable = True

        if i < nb_conv:
            norm_layer = model.get_layer('norm' + str(i))
            norm_layer.trainable = True
            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    layer = model.layers[-3]  # last convolutional layer
    print(layer.name)
    layer.trainable = True
    weights = layer.get_weights()
    print(np.shape(weights))

    new_kernel = np.random.normal(size=weights[0].shape) / (GRIDSZ * GRIDSZ)
    new_bias = np.random.normal(size=weights[1].shape) / (GRIDSZ * GRIDSZ)

    print(model.layers[0].name)
    print(model(x))