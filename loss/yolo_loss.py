# ================================================================
#
#   Editor      : Pycharm
#   File name   : yolo_loss
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:58
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 计算真实值和预测值的误差
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

import tensorflow as tf
import numpy as np
from models.darknet19 import darknet_Model
from models.yolo import yolo_Model
from utils.groud_dataset import Dataloader
# 计算当前ancorebox和gtbox比,训练的框和锚点的框的iou
def compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    # x:[b,16,16,5]
    xmin1 = x1 - 0.5 * w1
    ymin1 = y1 - 0.5 * h1  # 锚点真实值左上角坐标
    xmax1 = x1 + 0.5 * w1
    ymax1 = y1 + 0.5 * h1  # 锚点真实值右下角坐标

    xmin2 = x2 - 0.5 * w2
    ymin2 = y2 - 0.5 * h2  # 锚点预测值左上角坐标
    xmax2 = x2 + 0.5 * w2
    ymax2 = y2 + 0.5 * h2  # 锚点预测值右下角坐标

    interw = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interw * interh
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-6)

    # iou:[b,16,16,5]
    return iou

#计算误差(yolo算法精髓)
def yolo_loss(bath_detector_mast, bath_matching_gt_box,matching_classes_oh , bath_gt_boxes_grid, y_pred,GRIDSZ,ANCHORS):
    # bath_detector_mast(G, G, 3, 1)
    # bath_matching_gt_box(G, G, 3, 5)
    # matching_classes_oh(G, G, 3, 6)
    # bath_gt_boxes_grid(6, 5)
    # y_pred(1, G, G, 3, 11)
    # 思路分析:
    # 通过模型进行一次正向传播以后,会得到一个[b,16,16,a,5+n]的传播数据,我们先考虑如何得到x和y的误差
    # 由于模型正向传播后,x和y的范围是[-∞,+∞],因此我们可以先创建一个16*16的坐标网格点,然后将xysigmoid后加上对应的网格点,就可以得到
    # 每个锚点在对应的小方格内的坐标,不用(1+x,1+y)后与真实不符,会通过训练把其变为0或者接近真实值,只用考虑其误差
    # 由于我们找的是5个锚点的对应网格点,所以网格点格式应该是[b,16,16,a,2]这里的2指的就是xy

    y_pred=tf.squeeze(y_pred)
    # 创建坐标轴
    bath_detector_mast=tf.cast(bath_detector_mast,tf.float32)
    bath_matching_gt_box= tf.cast(bath_matching_gt_box, tf.float32)
    matching_classes_oh= tf.cast(matching_classes_oh, tf.float32)
    bath_gt_boxes_grid = tf.cast(bath_gt_boxes_grid, tf.float32)
    # print("GRIDSZ",GRIDSZ)
    # print("ANCHORS",ANCHORS)
    #anchor_exit=tf.reduce_sum(bath_detector_mast)
    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])  # [256,]
    x_grid = tf.reshape(x_grid, (GRIDSZ, GRIDSZ, 1, 1))  # [1,16,16,1,1]
    x_grid = tf.cast(x_grid, tf.float32)

    y_grid = tf.transpose(x_grid, [1, 0, 2, 3])  # [1,16,16,1,1],坐标轴里xy相反
    xy_grid = tf.concat([x_grid, y_grid], axis=-1)  # [1,16,16,1,2]
    xy_grid = tf.tile(xy_grid, [1, 1, 3, 1])  # [b,16,16,a,2]
    xy_grid = tf.cast(xy_grid, tf.float32)
    # 获得xy绝对坐标(锚点在每个方格的坐标)
    xy_pred = y_pred[..., 0:2]  # [b,16,16,a,2]
    xy_pred = tf.sigmoid(xy_pred)
    xy_pred = xy_pred + xy_grid

    # 统计总共有几个被选框(一张图里有几个框)
    n_detector_mask = tf.reduce_sum(tf.cast(bath_detector_mast > 0., tf.float32))

    # 计算xyloss(MSE算法)

    xy_loss = (bath_detector_mast * tf.square(bath_matching_gt_box[..., 0:2] - xy_pred)) / (
            n_detector_mask + 1e-6)  # [1,16,16,a,2]
    xy_loss = tf.reduce_sum(xy_loss)
    # print("xyloss",xy_loss)
    # 锚点wh和预测wh相乘得到相对wh
    anchors = np.array(ANCHORS).reshape(3, 2)
    wh_pred = tf.exp(y_pred[..., 2:4])
    #wh_pred = tf.sigmoid(y_pred[..., 2:4])
    # print("预测宽度数据", bath_detector_mast * wh_pred)  # [6.5277815 9.200164 ]
    #print("wh:",np.shape(wh_pred))
    #print("a:",anchors)
    wh_pred = wh_pred * anchors

    #demo
    # wh_pred
    # print(bath_detector_mast *bath_matching_gt_box[...,2:4])# [5.4375 7.    ]
    # print("预测宽度数据",bath_detector_mast *wh_pred)#[6.5277815 9.200164 ]
    # print("预测宽度数据",bath_detector_mast *tf.sqrt(wh_pred))#[1.6599464 4.062557 ]
    # print("实际宽度,",bath_detector_mast *tf.sqrt(bath_matching_gt_box[...,2:4]))#[2.3318448 2.6457512]

    # 计算whloss(MSE算法,注意要将开跟,因为前面两个Wh相乘)
    # wh_loss = bath_detector_mast * tf.square(tf.sqrt(bath_matching_gt_box[..., 2:4]) - tf.sqrt(wh_pred)) / (
    #         n_detector_mask + 1e-6)  # [1,16,16,5,2]
    wh_loss = bath_detector_mast * tf.square(tf.sqrt(bath_matching_gt_box[..., 2:4]) - tf.sqrt((wh_pred))) / (
            n_detector_mask + 1e-6)
    wh_loss = tf.reduce_sum(wh_loss)

    coord_loss = (xy_loss + wh_loss)

    # 根据概率找出最合适的标签
    class_pred = y_pred[..., 5:]  # [b,16,16,a,2]
    class_true = tf.argmax(matching_classes_oh, axis=-1)  # [b,16,16,a]

    # 计算classloss(交叉熵)
    class_loss = tf.losses.sparse_categorical_crossentropy(class_true, class_pred, from_logits=True)  # [b,16,16,a]
    class_loss = tf.expand_dims(class_loss, -1) * bath_detector_mast  # [b,16,16,a]->[b,16,16,a,1]*[b16,16,a,1]
    class_loss = tf.reduce_sum(class_loss) / (n_detector_mask + 1e-6)
    # print("classloss", class_loss)

    # object loss
    # 获取锚点真实值x1,y1,w1,h1:[b,16,16,a]
    x1, y1, w1, h1 = bath_matching_gt_box[..., 0], bath_matching_gt_box[..., 1], bath_matching_gt_box[..., 2], \
                     bath_matching_gt_box[..., 3]
    # 获取锚点预测值x2,y2,w2,h2:[b,16,16,a]
    x2, y2, w2, h2 = xy_pred[..., 0], xy_pred[..., 1], wh_pred[..., 0], wh_pred[..., 1]
    # 计算锚点和预测iou
    ious = compute_iou(x1, y1, w1, h1, x2, y2, w2, h2)
    ious = tf.expand_dims(ious, axis=-1)  # [b,16,16,a,1]

    # [b,16,16,a,1]类别
    pred_conf = tf.sigmoid(y_pred[..., 4:5])

    # 思路分析:因为预测的框是[b, 16, 16, a, 5]要和真实框[b,40,a]进行iou的话,两边都要改格式
    # 初始化标记框[b,40,5]->[b,1,1,1,40,a]
    box_true = tf.reshape(bath_gt_boxes_grid, \
                          [ 1, 1, 1, bath_gt_boxes_grid.shape[0],
                           bath_gt_boxes_grid.shape[1]])
    # 后期回顾不要晕,由于前面工作,已经把预测框[b,16,16,a,5]分xy和wh[b,16,16,a,2]
    # 初始化预测框[b,16,16,a,2]->[b,16,16,a,1,2] 这里分开个1其实相当于boxture里的40
    xy_true = box_true[..., 0:2]
    wh_true = box_true[..., 2:4]
    xy_pred = tf.expand_dims(xy_pred, axis=3)
    wh_pred = tf.expand_dims(wh_pred, axis=3)

    # 获取两个框左下和右下坐标
    xy_true_min = xy_true - 0.5 * wh_true
    xy_true_max = xy_true + 0.5 * wh_true
    xy_pred_min = xy_pred - 0.5 * wh_pred
    xy_pred_max = xy_pred + 0.5 * wh_pred

    # 计算IoU  这里写麻烦了,如果有机会,用def按照com_iou格式写看看
    # 这里用和0比较,是因为若两个框不相交,结果必然是负的,那么wh就是0
    # 这里有一个广播原则,pre是[b,16,16,a,1,2],true是[b,1,1,1,40,2] 所以得到的结果必然是[b,16,16,a,40,2]
    intersect_wh = tf.maximum(tf.minimum(xy_pred_max, xy_true_max) - tf.maximum(xy_pred_min, xy_true_min),
                              0.)  # [b,16,16,5,40,2]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # [b,16,16,5,1]
    # [b,16,16,a,1]+[b,1,1,1,40]-[b,16,16,a,40]=>[b,16,16,a,40] 不要晕,wh_p是[b,1,1,1,40,2]但是乘后变成[b,1,1,1,40]
    union_area = wh_pred[..., 0] * wh_pred[..., 1] + wh_true[..., 0] * wh_true[..., 1] - intersect_area

    # 此iou是标记款和预测框的iou,做这个iou,是为了让一些预测框和标记款的Iou小于0.6属于无物体
    # 比如这个地方预测出物体了,但是框里真实很远,iou<0.6,那么就当作没预测出物体来计算
    iou_score = intersect_area / union_area  # [b,16,16,a,40]
    best_iou = tf.reduce_max(iou_score, axis=3)
    # [b,16,16,5,1]
    best_iou = tf.expand_dims(best_iou, axis=-1)
    # 统计有多少个iou<0.6
    nonobj_detection_iou = tf.cast(best_iou < 0.6, tf.float32)
    # 判断哪些是nonobj,与之相反的是bath_detector_mask,判断条件(与detector相反,nonobj_detection=0)
    nonobj_detection = nonobj_detection_iou * (1 - bath_detector_mast)
    n_nonobj = tf.reduce_sum(tf.cast(nonobj_detection > 0., tf.float32))
    # 计算nonobj_loss
    nonobj_loss = tf.reduce_sum(nonobj_detection * tf.square(-pred_conf)) / (n_nonobj + 1e-6)
    # 计算obj_loss
    obj_loss = tf.reduce_sum(bath_detector_mast * tf.square(ious - pred_conf)) / (n_detector_mask + 1e-6)
    loss = (coord_loss + class_loss + 5* obj_loss + nonobj_loss)
    #loss = (class_loss)


    #return loss, [nonobj_loss + 5 * obj_loss, class_loss, coord_loss]
    sub_loss=[float(nonobj_loss) + 5 * float(obj_loss), float(class_loss), float(coord_loss)]

    # if coord_loss==0:
    #     loss=0
    return loss,sub_loss

# 测试loss
def demo_loss(db,model,GRIDSZ,ANCHORS,ANCHORS_N):
    img, bath_detector_mask, bath_matching_gt_box, bath_matching_classes_oh, bath_gt_boxes_grid= next(db)

    y_preds = model(tf.expand_dims(img[0], axis=0))
    loss = 0
    for b in range(img.shape[0]):
        for a in range(3):
            anchors_loss=yolo_loss(bath_detector_mask[b][a],
                      bath_matching_gt_box[b][a],
                      bath_matching_classes_oh[b][a],
                      bath_gt_boxes_grid[b][a],
                      y_preds[a],
                      GRIDSZ[a],ANCHORS[a],ANCHORS_N)
        loss += anchors_loss / 3
    print("loss:", loss)


if __name__ == '__main__':
    obj_names = ('normal','anthracnose','blight','insect_attack','leaf_blight','powdery_mildew')
    ANCHORS = [[0.19417476, 0.25242718,
                0.31067961, 0.58252427,
                0.6407767, 0.44660194],
               [
                   1.16504854, 2.36893204,
                   2.40776699, 1.74757282,
                   2.29126214, 4.62135922],

               [9.00970874, 6.99029126,
                12.11650485, 15.37864078,
                28.97087379, 25.32038835]
               ]
    IMGSZ = 256
    GRIDSZ = [8,16,32]
    ANCHORS_N= 3
    # 读取数据
    dataloader = Dataloader(r'..\data\watermelon\test1\image', r'..\data\watermelon\test1\annotation', 5, obj_names,
                            ANCHORS, IMGSZ, GRIDSZ)
    gen = dataloader()
    model = yolo_Model()
    model.build(input_shape=(None, IMGSZ, IMGSZ, 3))
    demo_loss(gen,model,GRIDSZ,ANCHORS,ANCHORS_N)