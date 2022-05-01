# ================================================================
#
#   Editor      : Pycharm
#   File name   : yolo
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:58
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : yolo网络
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np



#模型
class yolo_Model(tf.keras.Model):
    def __init__(self,numwork=6):
        super(yolo_Model, self).__init__()
        self.numwork = numwork
        ######### 2 2
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', name='conv1', use_bias=False)
        self.bn1 = layers.BatchNormalization(name='norm1')

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', name='conv2', use_bias=False)
        self.bn2 = layers.BatchNormalization(name='norm2')
        #########  2 4
        self.conv3 = layers.Conv2D(32, (1, 1), padding='same', name='conv3', use_bias=False)
        self.bn3 = layers.BatchNormalization(name='norm3')

        self.conv4 = layers.Conv2D(64, (3, 3), padding='same', name='conv4', use_bias=False)
        self.bn4 = layers.BatchNormalization(name='norm4')
        ######## 1 5
        self.conv5 = layers.Conv2D(128, (3, 3), padding='same', name='conv5', use_bias=False)
        self.bn5 = layers.BatchNormalization(name='norm5')
        ######## 4 9
        self.conv6 = layers.Conv2D(64, (1, 1), padding='same', name='conv6', use_bias=False)
        self.bn6 = layers.BatchNormalization(name='norm6')

        self.conv7 = layers.Conv2D(128, (3, 3), padding='same', name='conv7', use_bias=False)
        self.bn7 = layers.BatchNormalization(name='norm7')

        self.conv8 = layers.Conv2D(64, (1, 1), padding='same', name='conv8', use_bias=False)
        self.bn8 = layers.BatchNormalization(name='norm8')

        self.conv9 = layers.Conv2D(128, (3, 3), padding='same', name='conv9', use_bias=False)
        self.bn9 = layers.BatchNormalization(name='norm9')
        ######### 1 10
        self.conv10 = layers.Conv2D(256, (3, 3), padding='same', name='conv10', use_bias=False)
        self.bn10 = layers.BatchNormalization(name='norm10')
        ######### 16 26
        self.conv11 = layers.Conv2D(128, (1, 1), padding='same', name='conv11', use_bias=False)
        self.bn11 = layers.BatchNormalization(name='norm11')

        self.conv12 = layers.Conv2D(256, (3, 3), padding='same', name='conv12', use_bias=False)
        self.bn12 = layers.BatchNormalization(name='norm12')

        self.conv13 = layers.Conv2D(128, (1, 1), padding='same', name='conv13', use_bias=False)
        self.bn13 = layers.BatchNormalization(name='norm13')

        self.conv14 = layers.Conv2D(256, (3, 3), padding='same', name='conv14', use_bias=False)
        self.bn14 = layers.BatchNormalization(name='norm14')

        self.conv15 = layers.Conv2D(128, (1, 1), padding='same', name='conv15', use_bias=False)
        self.bn15 = layers.BatchNormalization(name='norm15')

        self.conv16 = layers.Conv2D(256, (3, 3), padding='same', name='conv16', use_bias=False)
        self.bn16 = layers.BatchNormalization(name='norm16')

        self.conv17 = layers.Conv2D(128, (1, 1), padding='same', name='conv17', use_bias=False)
        self.bn17 = layers.BatchNormalization(name='norm17')

        self.conv18 = layers.Conv2D(256, (3, 3), padding='same', name='conv18', use_bias=False)
        self.bn18 = layers.BatchNormalization(name='norm18')

        self.conv19 = layers.Conv2D(128, (1, 1), padding='same', name='conv19', use_bias=False)
        self.bn19 = layers.BatchNormalization(name='norm19')

        self.conv20 = layers.Conv2D(256, (3, 3), padding='same', name='conv20', use_bias=False)
        self.bn20 = layers.BatchNormalization(name='norm20')

        self.conv21 = layers.Conv2D(128, (1, 1), padding='same', name='conv21', use_bias=False)
        self.bn21 = layers.BatchNormalization(name='norm21')

        self.conv22 = layers.Conv2D(256, (3, 3), padding='same', name='conv22', use_bias=False)
        self.bn22 = layers.BatchNormalization(name='norm22')

        self.conv23 = layers.Conv2D(128, (1, 1), padding='same', name='conv23', use_bias=False)
        self.bn23 = layers.BatchNormalization(name='norm23')

        self.conv24 = layers.Conv2D(256, (3, 3), padding='same', name='conv24', use_bias=False)
        self.bn24 = layers.BatchNormalization(name='norm24')

        self.conv25 = layers.Conv2D(128, (1, 1), padding='same', name='conv25', use_bias=False)
        self.bn25 = layers.BatchNormalization(name='norm25')

        self.conv26 = layers.Conv2D(256, (3, 3), padding='same', name='conv26', use_bias=False)
        self.bn26 = layers.BatchNormalization(name='norm26')
        ######## 1 27

        self.conv27 = layers.Conv2D(512, (3, 3), padding='same', name='conv27', use_bias=False)
        self.bn27 = layers.BatchNormalization(name='norm27')

        ####### 16 43

        self.conv28 = layers.Conv2D(256, (1, 1), padding='same', name='conv28', use_bias=False)
        self.bn28 = layers.BatchNormalization(name='norm28')

        self.conv29 = layers.Conv2D(512, (3, 3), padding='same', name='conv29', use_bias=False)
        self.bn29 = layers.BatchNormalization(name='norm29')

        self.conv30 = layers.Conv2D(256, (1, 1), padding='same', name='conv30', use_bias=False)
        self.bn30 = layers.BatchNormalization(name='norm30')

        self.conv31 = layers.Conv2D(512, (3, 3), padding='same', name='conv31', use_bias=False)
        self.bn31 = layers.BatchNormalization(name='norm31')

        self.conv32 = layers.Conv2D(256, (1, 1), padding='same', name='conv32', use_bias=False)
        self.bn32 = layers.BatchNormalization(name='norm32')

        self.conv33 = layers.Conv2D(512, (3, 3), padding='same', name='conv33', use_bias=False)
        self.bn33 = layers.BatchNormalization(name='norm33')

        self.conv34 = layers.Conv2D(256, (1, 1), padding='same', name='conv34', use_bias=False)
        self.bn34 = layers.BatchNormalization(name='norm34')

        self.conv35 = layers.Conv2D(512, (3, 3), padding='same', name='conv35', use_bias=False)
        self.bn35 = layers.BatchNormalization(name='norm35')

        self.conv36 = layers.Conv2D(256, (1, 1), padding='same', name='conv36', use_bias=False)
        self.bn36 = layers.BatchNormalization(name='norm36')

        self.conv37 = layers.Conv2D(512, (3, 3), padding='same', name='conv37', use_bias=False)
        self.bn37 = layers.BatchNormalization(name='norm37')

        self.conv38 = layers.Conv2D(256, (1, 1), padding='same', name='conv38', use_bias=False)
        self.bn38 = layers.BatchNormalization(name='norm38')

        self.conv39 = layers.Conv2D(512, (3, 3), padding='same', name='conv39', use_bias=False)
        self.bn39 = layers.BatchNormalization(name='norm39')

        self.conv40 = layers.Conv2D(256, (1, 1), padding='same', name='conv40', use_bias=False)
        self.bn40 = layers.BatchNormalization(name='norm40')

        self.conv41 = layers.Conv2D(512, (3, 3), padding='same', name='conv41', use_bias=False)
        self.bn41 = layers.BatchNormalization(name='norm41')

        self.conv42 = layers.Conv2D(256, (1, 1), padding='same', name='conv42', use_bias=False)
        self.bn42 = layers.BatchNormalization(name='norm42')

        self.conv43 = layers.Conv2D(512, (3, 3), padding='same', name='conv43', use_bias=False)
        self.bn43 = layers.BatchNormalization(name='norm43')

        ###### 1 44
        self.conv44 = layers.Conv2D(1024, (3, 3), padding='same', name='conv44', use_bias=False)
        self.bn44 = layers.BatchNormalization(name='norm44')

        ####### 16 52

        self.conv45 = layers.Conv2D(512, (1, 1), padding='same', name='conv45', use_bias=False)
        self.bn45 = layers.BatchNormalization(name='norm45')

        self.conv46 = layers.Conv2D(1024, (3, 3), padding='same', name='conv46', use_bias=False)
        self.bn46 = layers.BatchNormalization(name='norm46')

        self.conv47 = layers.Conv2D(512, (1, 1), padding='same', name='conv47', use_bias=False)
        self.bn47 = layers.BatchNormalization(name='norm47')

        self.conv48 = layers.Conv2D(1024, (3, 3), padding='same', name='conv48', use_bias=False)
        self.bn48 = layers.BatchNormalization(name='norm48')

        self.conv49 = layers.Conv2D(512, (1, 1), padding='same', name='conv49', use_bias=False)
        self.bn49 = layers.BatchNormalization(name='norm49')

        self.conv50 = layers.Conv2D(1024, (3, 3), padding='same', name='conv50', use_bias=False)
        self.bn50 = layers.BatchNormalization(name='norm50')

        self.conv51 = layers.Conv2D(512, (1, 1), padding='same', name='conv51', use_bias=False)
        self.bn51 = layers.BatchNormalization(name='norm51')

        self.conv52 = layers.Conv2D(1024, (3, 3), padding='same', name='conv52', use_bias=False)
        self.bn52 = layers.BatchNormalization(name='norm52')

        # self.conv53 = layers.Conv2D(3 * (5 + self.numwork), (1, 1), padding='same', name='conv53', use_bias=False)
        self.conv53 = layers.Conv2D(3 * (5 + self.numwork), (1, 1), padding='same', name='conv53')
        self.conv54 = layers.Conv2D(3 * (5 + self.numwork), (1, 1), padding='same', name='conv54')
        self.conv55 = layers.Conv2D(3 * (5 + self.numwork), (1, 1), padding='same', name='conv55')

        self.maxpool = layers.MaxPooling2D(pool_size=(2, 2))

        self.leakrelu = layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, training=None, mask=None):
        # print("input:",np.shape(inputs))

        # unit 1 2 1-2
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.leakrelu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # print("lay1:", np.shape(x))

        # unit 2 2*1 3-4
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.leakrelu(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.leakrelu(x)
        # print("lay2:", np.shape(x))

        # unit 3 1 5
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # print("lay3:", np.shape(x))

        # unit 4 2*2 6-9
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.leakrelu(x)
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.leakrelu(x)
        x = self.conv8(x)
        x = self.bn8(x, training=training)
        x = self.leakrelu(x)
        x = self.conv9(x)
        x = self.bn9(x, training=training)
        x = self.leakrelu(x)
        # print("lay4:", np.shape(x))

        # unit 5 1 10
        x = self.conv10(x)
        x = self.bn10(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # print("lay5:", np.shape(x))

        # unit 6 2*8 11-26
        x = self.conv11(x)
        x = self.bn11(x, training=training)
        x = self.leakrelu(x)
        x = self.conv12(x)
        x = self.bn12(x, training=training)
        x = self.leakrelu(x)
        x = self.conv13(x)
        x = self.bn13(x, training=training)
        x = self.leakrelu(x)
        x = self.conv14(x)
        x = self.bn14(x, training=training)
        x = self.leakrelu(x)
        x = self.conv15(x)
        x = self.bn15(x, training=training)
        x = self.leakrelu(x)
        x = self.conv16(x)
        x = self.bn16(x, training=training)
        x = self.leakrelu(x)
        x = self.conv17(x)
        x = self.bn17(x, training=training)
        x = self.leakrelu(x)
        x = self.conv18(x)
        x = self.bn18(x, training=training)
        x = self.leakrelu(x)
        x = self.conv19(x)
        x = self.bn19(x, training=training)
        x = self.leakrelu(x)
        x = self.conv20(x)
        x = self.bn20(x, training=training)
        x = self.leakrelu(x)
        x = self.conv21(x)
        x = self.bn21(x, training=training)
        x = self.leakrelu(x)
        x = self.conv22(x)
        x = self.bn22(x, training=training)
        x = self.leakrelu(x)
        x = self.conv23(x)
        x = self.bn23(x, training=training)
        x = self.leakrelu(x)
        x = self.conv24(x)
        x = self.bn24(x, training=training)
        x = self.leakrelu(x)
        x = self.conv25(x)
        x = self.bn25(x, training=training)
        x = self.leakrelu(x)
        x = self.conv26(x)
        x = self.bn26(x, training=training)
        x = self.leakrelu(x)

        skip_x_1 = x
        # print("lay6:", np.shape(x))

        # unit 7 1 27
        x = self.conv27(x)
        x = self.bn27(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # print("lay7:", np.shape(x))

        # unit 8 2*8 28-43
        x = self.conv28(x)
        x = self.bn28(x, training=training)
        x = self.leakrelu(x)
        x = self.conv29(x)
        x = self.bn29(x, training=training)
        x = self.leakrelu(x)
        x = self.conv30(x)
        x = self.bn30(x, training=training)
        x = self.leakrelu(x)
        x = self.conv31(x)
        x = self.bn31(x, training=training)
        x = self.leakrelu(x)
        x = self.conv32(x)
        x = self.bn32(x, training=training)
        x = self.leakrelu(x)
        x = self.conv33(x)
        x = self.bn33(x, training=training)
        x = self.leakrelu(x)
        x = self.conv34(x)
        x = self.bn34(x, training=training)
        x = self.leakrelu(x)
        x = self.conv35(x)
        x = self.bn35(x, training=training)
        x = self.leakrelu(x)
        x = self.conv36(x)
        x = self.bn36(x, training=training)
        x = self.leakrelu(x)
        x = self.conv37(x)
        x = self.bn37(x, training=training)
        x = self.leakrelu(x)
        x = self.conv38(x)
        x = self.bn38(x, training=training)
        x = self.leakrelu(x)
        x = self.conv39(x)
        x = self.bn39(x, training=training)
        x = self.leakrelu(x)
        x = self.conv40(x)
        x = self.bn40(x, training=training)
        x = self.leakrelu(x)
        x = self.conv41(x)
        x = self.bn41(x, training=training)
        x = self.leakrelu(x)
        x = self.conv42(x)
        x = self.bn42(x, training=training)
        x = self.leakrelu(x)
        x = self.conv43(x)
        x = self.bn43(x, training=training)
        x = self.leakrelu(x)

        skip_x_2 = x
        # print("lay8:", np.shape(x))

        # unit 9 1 44
        x = self.conv44(x)
        x = self.bn44(x, training=training)
        x = self.leakrelu(x)
        x = self.maxpool(x)
        # print("lay9:", np.shape(x))

        # unit 10 8 2*4 45-53
        x = self.conv45(x)
        x = self.bn45(x, training=training)
        x = self.leakrelu(x)
        x = self.conv46(x)
        x = self.bn46(x, training=training)
        x = self.leakrelu(x)
        x = self.conv47(x)
        x = self.bn47(x, training=training)
        x = self.leakrelu(x)
        x = self.conv48(x)
        x = self.bn48(x, training=training)
        x = self.leakrelu(x)
        x = self.conv49(x)
        x = self.bn49(x, training=training)
        x = self.leakrelu(x)
        x = self.conv50(x)
        x = self.bn50(x, training=training)
        x = self.leakrelu(x)
        x = self.conv51(x)
        x = self.bn51(x, training=training)
        x = self.leakrelu(x)
        x = self.conv52(x)
        x = self.bn52(x, training=training)
        x = self.leakrelu(x)
        skip_x_3 = x
        # print("lay10:", np.shape(x))

        skip_x_2 = tf.concat([self.maxpool(skip_x_1), skip_x_2], axis=-1)
        skip_x_3 = tf.concat([self.maxpool(skip_x_2), skip_x_3], axis=-1)

        out1 = self.conv53(skip_x_1)
        out2 = self.conv54(skip_x_2)
        out3 = self.conv55(skip_x_3)

        out1 = layers.Reshape([32, 32, 3, (5 + self.numwork)])(out1)
        out2 = layers.Reshape([16, 16, 3, (5 + self.numwork)])(out2)
        out3 = layers.Reshape([8, 8, 3, (5 + self.numwork)])(out3)

        return [out3, out2, out1]

#模型初始化
class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4

    def __call__(self,model,GRIDSZ,*args, **kwargs):
        self.reset()
        nb_conv = 52

        for i in range(1, nb_conv + 1):
            conv_layer = model.get_layer('conv' + str(i))
            conv_layer.trainable = True

            if i < nb_conv:
                norm_layer = model.get_layer('norm' + str(i))
                norm_layer.trainable = True
                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = self.read_bytes(size)
                gamma = self.read_bytes(size)
                mean = self.read_bytes(size)
                var = self.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights()) > 1:
                bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                #print(conv_layer.name)
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])

        for i in range(3):
            layer = model.layers[-3 - i]  # last convolutional layer
            layer.trainable = True

            weights = layer.get_weights()

            new_kernel = np.random.normal(size=weights[0].shape) / (GRIDSZ[i] * GRIDSZ[i])
            new_bias = np.random.normal(size=weights[1].shape) / (GRIDSZ[i] * GRIDSZ[i])

            layer.set_weights([new_kernel, new_bias])
        return model
if __name__ == '__main__':
    x = tf.random.normal([3, 256, 256, 3])
    model = yolo_Model()
    model(x)
    weight_reader = WeightReader('../yolo.weights')
    GRIDSZ = [8, 16, 32]
    model=weight_reader(model,GRIDSZ)

    #
    # print(np.shape(model(x)[0]))
    # print(np.shape(model(x)[1]))
    # print(np.shape(model(x)[2]))