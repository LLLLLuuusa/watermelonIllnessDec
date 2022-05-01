# ================================================================
#
#   Editor      : Pycharm
#   File name   : train
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 14:05
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 深度学习训练
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================


from tensorflow.keras import optimizers
from models.darknet19 import darknet_Model
from models.yolo2 import *
from utils.parse_config import parse_data_cfg
from loss.yolo_loss import yolo_loss
from utils.groud_dataset import Dataloader
import tensorflow as tf
from tqdm import trange
import time


class train():

    def __init__(self,cfg_path):

        #读取配置文件信息
        self.get_data_cfg = parse_data_cfg(cfg_path)  # 返回训练配置参数，类型：字典
        self.IMGSZ = int(self.get_data_cfg['IMGSZ'])   # 图片大小
        self.lr = float(self.get_data_cfg['lr'])  # 学习率
        self.weight_paht = self.get_data_cfg["weight_path"] # 模型加载及保存的路径
        self.epoch = int(self.get_data_cfg["epoch"])    # 训练次数
        self.bath = int(self.get_data_cfg["bath"])  # 多线程处理
        self.GRIDSZ = tuple(eval(self.get_data_cfg['GRIDSZ']))      # 特征图大小
        self.ANCHORS = tuple(eval(self.get_data_cfg['ANCHORS']))    # 多尺度检测模型(?,我不确定中文是不是叫这个)
        self.obj_names = tuple(eval(self.get_data_cfg['obj_names'])) # 类别
        self.train_path = tuple(eval(self.get_data_cfg['train_Path'])) # 训练集模型路径
        self.test_path = tuple(eval(self.get_data_cfg['test_Path']))  # 验证集模型路径
        self.val_path = tuple(eval(self.get_data_cfg['val_Path']))  # 测试集模型路径
        self.writer = tf.summary.create_file_writer("log")  # 生成一个 TensorBoard 的输出

        # 模型
        self.model = yolo_Model()
        #self.model.load_weights(self.weight_paht)
        x = tf.random.normal([3, 256, 256, 3])
        self.model(x)
        # weight_reader = WeightReader('yolo.weights')
        # self.model=weight_reader(self.model,self.GRIDSZ)

        #开始训练!
        self.train()


    # 训练
    def train(self):
        # #加载数据集
        dataloader=Dataloader(self.train_path[0],self.train_path[1],self.bath,self.obj_names,self.ANCHORS,self.GRIDSZ,self.IMGSZ)
        gen=dataloader()

        #设置学习率优化器(Adam)
        optimizer = optimizers.Adam(learning_rate=self.lr, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08,clipnorm=1)
        #训练
        for epoch in range(self.epoch):
            with trange(int(len(dataloader)/self.bath)) as t:
                for step in t:
                    img, bath_detector_mask, bath_matching_gt_box, bath_matching_classes_oh, bath_gt_boxes_grid = next(gen)

                    with tf.GradientTape() as tape:
                        y_preds = self.model(img, training=True)
                        loss,sub_loss = 0,[0,0,0]
                        for b in range(img.shape[0]):
                            bath_loss,bath_sub_loss = 0,[0,0,0]
                            for a in range(3):
                                anchors_loss ,anchors_sub_loss= yolo_loss(bath_detector_mask[b][a],
                                                         bath_matching_gt_box[b][a],
                                                         bath_matching_classes_oh[b][a],
                                                         bath_gt_boxes_grid[b][a],
                                                         y_preds[a][b],
                                                         self.GRIDSZ[a], self.ANCHORS[a])
                                bath_loss += anchors_loss
                                bath_sub_loss[0]+=anchors_sub_loss[0]
                                bath_sub_loss[1] += anchors_sub_loss[1]
                                bath_sub_loss[2] += anchors_sub_loss[2]
                            loss += bath_loss
                            sub_loss[0] += bath_sub_loss[0]
                            sub_loss[1] += bath_sub_loss[1]
                            sub_loss[2] += bath_sub_loss[2]
                        loss = loss / img.shape[0]
                    print(loss)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    print(grads)
                    # grads, global_norm = tf.clip_by_global_norm(grads, 2)#梯度裁剪,有时候会梯度爆炸不知道为啥,直接给剪了一劳永逸
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    # 设置进度条右边显示的信息
                    des = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())) + " Epoch {}".format(epoch)
                    post = f"lr: {self.lr:.8f} total_loss: {loss:.2f} obj_loss: {sub_loss[0]:.2f} class_loss: {sub_loss[1]:.2f} coord_loss: {sub_loss[2]:.2f}"
                    t.set_description(des)
                    t.set_postfix_str(post)

            # 将本次epoch训练结果打印到日志中
            with self.writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=step)
                tf.summary.scalar("loss/total_loss", loss, step=step)
                tf.summary.scalar("loss/obj_loss", sub_loss[0], step=step)
                tf.summary.scalar("loss/class_loss", sub_loss[1], step=step)
                tf.summary.scalar("loss/coord_loss", sub_loss[2], step=step)
            self.writer.flush()

        #保存模型
        self.model.save_weights(self.weight_paht)

if __name__ == '__main__':
    train("cfg/watermelon.data")
