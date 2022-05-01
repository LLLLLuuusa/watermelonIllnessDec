# ================================================================
#
#   Editor      : Pycharm
#   File name   : preprocess
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:58
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 数据增强
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================
import tensorflow as tf

# 预处理格式
def preprocess(imgs, boxs):
    # imgs指路径
    # boxs指bndbox的坐标和标签
    x = tf.io.read_file(imgs)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x, boxs