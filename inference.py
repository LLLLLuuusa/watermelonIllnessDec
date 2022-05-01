# ================================================================
#
#   Editor      : Pycharm
#   File name   : inference
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:57
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 模型推理
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

import glob
from matplotlib import pyplot as plt
from matplotlib import patches
from utils.parse_config import parse_data_cfg
from models.yolo import yolo_Model
import cv2
import tensorflow as tf
import numpy as np
from models.yolo import yolo_Model
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager


""""""

#图片打开
def visualize_result(img_path, model,data_cfg):
    # imgs_path指图片路径
    # model 模型

    #获取参数
    get_data_cfg = parse_data_cfg(data_cfg)  # 返回训练配置参数，类型：字典
    GRIDSZS = tuple(eval(get_data_cfg['GRIDSZ']))
    #GRIDSZ = [8, 16, 32]
    IMGSZ = int(get_data_cfg['IMGSZ'])
    ANCHORS = tuple(eval(get_data_cfg['ANCHORS']))
    obj_name = tuple(eval(get_data_cfg['obj_names']))
    num_work = len(obj_name)

    # 对imgs_path进行预处理,获取图片信息
    img = cv2.imread(img_path)  # [512,512,3]
    # 由于cv读取文件是BGR而不是传统RGB,所以这里做倒置
    img = img[..., ::-1] / IMGSZ  # [512,512,3]
    img = tf.cast(img, dtype=tf.float32)  # 这里升纬是因为模型就是需要输入4纬
    img = tf.expand_dims(img, axis=0)  # [1,512,512,3]

    # 获取训练结果
    y_preds = model(img,training=False)

    # 画框
    # 设置画板
    fig, ax = plt.subplots(1, figsize=(10, 10))
    all_boxes=[]
    all_scores=[]
    all_classes=[]
    for a in range(len(GRIDSZS)):
        GRIDSZ=GRIDSZS[a]
        y_pred=y_preds[a]
        anchors = ANCHORS[a]
        # 创建坐标轴
        x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])  # [256,]
        x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))  # [1,16,16,1,1]
        x_grid = tf.cast(x_grid, tf.float32)
        y_grid = tf.transpose(x_grid, [0, 2, 1, 3, 4])  # [1,16,16,1,1],坐标轴里xy相反
        xy_grid = tf.concat([x_grid, y_grid], axis=-1)  # [1,16,16,1,2]
        #xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1, 1, 5, 1])  # [b,16,16,5,2]
        xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1, 1, 3, 1])  # [b,16,16,5,2]
        xy_grid = tf.cast(xy_grid, tf.float32)

        # 对预测数据进行预处理
        # 获取预测xy
        # 获得xy绝对坐标(锚点在每个方格的坐标)
        xy_pred = y_pred[..., 0:2]  # [b,16,16,5,2]
        # print(xy_pred)
        xy_pred = tf.sigmoid(xy_pred)

        xy_pred = xy_pred + xy_grid
        # 获取预测wh
        # 锚点wh和预测wh相乘得到相对wh
        anchors = np.array(anchors).reshape(3, 2)
        wh_pred = tf.exp(y_pred[..., 2:4])
        wh_pred = wh_pred * anchors
        # 归一化0~1
        GRIDSZ_f16=float(GRIDSZ)
        xy_pred = xy_pred / tf.constant([GRIDSZ_f16, GRIDSZ_f16])
        wh_pred = wh_pred / tf.constant([GRIDSZ_f16, GRIDSZ_f16])
        # [b,16,16,5,1]类别
        pred_conf = tf.sigmoid(y_pred[..., 4:5])  # [1,16,16,5,1]
        # l1 l2类别概率
        pred_prob = tf.nn.softmax(y_pred[..., 5:])  # [1,16,16,5,2]
        # 降纬,xy[16,16,5,2],wh[16,16,5,2],conf[16,16,5],prob[16,16,5,2]
        xy_pred, wh_pred, pred_conf, pred_prob = \
            xy_pred[0], wh_pred[0], pred_conf[0], pred_prob[0]

        # 获取信息
        # 获取左上角坐标和右下角坐标
        boxes_xymin = xy_pred - 0.5 * wh_pred
        boxes_xymax = xy_pred + 0.5 * wh_pred
        boxes = tf.concat((boxes_xymin, boxes_xymax), axis=-1)  # [16,16,5,2+2]
        # 获取概率
        box_score = pred_conf * pred_prob  # [16,16,5,2]
        # print(box_score.shape,"score")
        # 获取类别  0是sugarbeet,1是weet,(score最大的为类别下标)
        box_class = tf.argmax(box_score, axis=-1)  # [16,16,5]
        # print(box_class.shape,"class")
        # 根据下标获取该类别概率(score最大的为类别概率)
        box_class_score = tf.reduce_max(box_score, axis=-1)  # [16,16,5]

        # 筛选信息
        # 判断范围,若概率过小不进行画框,返回布尔
        #print(box_class_score)

        pred_mask = box_class_score > 0.8 # 获取筛选过后的坐标
        boxes = tf.boolean_mask(boxes, pred_mask)  # [16,16,5,4]->[4]
        # 获取筛选过后的概率
        scores = tf.boolean_mask(box_class_score, pred_mask)  # [16,16,5] -> [N]
        # 获取筛选过后的类别
        classes = tf.boolean_mask(box_class, pred_mask)  # [16,16，5]-> [N]
        # 这里一定要写*IMGSZ,否则无法画
        boxes = boxes * IMGSZ

        if all_boxes==[]:
            all_boxes=boxes
            all_scores=scores
            all_classes=classes
        else:
            all_boxes=tf.concat([all_boxes,boxes],axis=0)
            all_scores = tf.concat([all_scores, scores], axis=0)
            all_classes = tf.concat([all_classes, classes], axis=0)
        print("完成一个box:", np.shape(all_boxes))
    print("未极大值抑制前框数为:",np.shape(all_boxes))
    # 非极大值抑制,防止画过多框,就根据scores进行限制,优先画分最大的,最多画40个
    select_idx = tf.image.non_max_suppression(all_boxes, all_scores, 40, iou_threshold=0.3)
    boxes = tf.gather(all_boxes, select_idx)  # [4]
    scores = tf.gather(all_scores, select_idx)  # [N]
    classes = tf.gather(all_classes, select_idx)  # [N]

    # 读取图片
    ax.imshow(img[0])
    # 总框数
    n_boxes = boxes.shape[0]
    print("总框数:",n_boxes)
    for i in range(n_boxes):

        # 坐标信息
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        label = classes[i].numpy()

        # 设置颜色
        if label == 0:  # sugarweet
            color = (0, 1, 0)
        else:
            color = (1, 0, 0)

        # 画矩形框
        rect = patches.Rectangle((x1.numpy(), y1.numpy()), w.numpy(), h.numpy(), linewidth=3, edgecolor=color,
                                 facecolor='none')

        # 概率
        label_name = {}
        for n in range(num_work):
            label_name[n] = obj_name[n]
        # ax.text(x1, y1, f"{label_name[label]}:{scores[i]:.1%}", bbox={'facecolor': color, 'alpha': 0.5})
        ax.text(x1, y1, f"{label_name[label]}:{scores[i]:.1%}", bbox={'facecolor': color, 'alpha': 0.5})
        # 添加
        ax.add_patch(rect)

#图片打开
def visualize_rectangle(img,model,data_cfg):
    # imgs_path指图片路径
    # model 模型
    #获取参数
    get_data_cfg = parse_data_cfg(data_cfg)  # 返回训练配置参数，类型：字典
    IMGSZ = int(get_data_cfg['IMGSZ'])
    weight_paht = get_data_cfg["weight_path"]
    GRIDSZS = tuple(eval(get_data_cfg['GRIDSZ']))
    ANCHORS = tuple(eval(get_data_cfg['ANCHORS']))
    obj_name = tuple(eval(get_data_cfg['obj_names']))
    num_work = len(obj_name)

    # model = yolo_Model()
    # # model = darknet_Model()
    # model.build(input_shape=(None, IMGSZ, IMGSZ, 3))
    # model.load_weights(weight_paht)

    # 对imgs_path进行预处理,获取图片信息
    #img = cv2.imread(img_path)  # [512,512,3]
    # 由于cv读取文件是BGR而不是传统RGB,所以这里做倒置
    img = img[..., ::-1] / 255.  # [512,512,3]
    img = tf.cast(img, dtype=tf.float32)  # 这里升纬是因为模型就是需要输入4纬
    img = tf.expand_dims(img, axis=0)  # [1,512,512,3]

    # 获取训练结果
    y_preds = model(img,training=False)

    all_boxes=[]
    all_scores=[]
    all_classes=[]
    for a in range(len(GRIDSZS)):
        GRIDSZ=GRIDSZS[a]
        y_pred=y_preds[a]
        anchors = ANCHORS[a]
        # 创建坐标轴
        x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])  # [256,]
        x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))  # [1,16,16,1,1]
        x_grid = tf.cast(x_grid, tf.float32)
        y_grid = tf.transpose(x_grid, [0, 2, 1, 3, 4])  # [1,16,16,1,1],坐标轴里xy相反
        xy_grid = tf.concat([x_grid, y_grid], axis=-1)  # [1,16,16,1,2]
        #xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1, 1, 5, 1])  # [b,16,16,5,2]
        xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1, 1, 3, 1])  # [b,16,16,5,2]
        xy_grid = tf.cast(xy_grid, tf.float32)

        # 对预测数据进行预处理
        # 获取预测xy
        # 获得xy绝对坐标(锚点在每个方格的坐标)
        xy_pred = y_pred[..., 0:2]  # [b,16,16,5,2]
        # print(xy_pred)
        xy_pred = tf.sigmoid(xy_pred)

        xy_pred = xy_pred + xy_grid
        # 获取预测wh
        # 锚点wh和预测wh相乘得到相对wh
        anchors = np.array(anchors).reshape(3, 2)
        wh_pred = tf.exp(y_pred[..., 2:4])
        wh_pred = wh_pred * anchors
        # 归一化0~1
        GRIDSZ_f16=float(GRIDSZ)
        xy_pred = xy_pred / tf.constant([GRIDSZ_f16, GRIDSZ_f16])
        wh_pred = wh_pred / tf.constant([GRIDSZ_f16, GRIDSZ_f16])
        # [b,16,16,5,1]类别
        pred_conf = tf.sigmoid(y_pred[..., 4:5])  # [1,16,16,5,1]
        # l1 l2类别概率
        pred_prob = tf.nn.softmax(y_pred[..., 5:])  # [1,16,16,5,2]
        # 降纬,xy[16,16,5,2],wh[16,16,5,2],conf[16,16,5],prob[16,16,5,2]
        xy_pred, wh_pred, pred_conf, pred_prob = \
            xy_pred[0], wh_pred[0], pred_conf[0], pred_prob[0]

        # 获取信息
        # 获取左上角坐标和右下角坐标
        boxes_xymin = xy_pred - 0.5 * wh_pred
        boxes_xymax = xy_pred + 0.5 * wh_pred
        boxes = tf.concat((boxes_xymin, boxes_xymax), axis=-1)  # [16,16,5,2+2]
        # 获取概率
        box_score = pred_conf * pred_prob  # [16,16,5,2]
        # print(box_score.shape,"score")
        # 获取类别  0是sugarbeet,1是weet,(score最大的为类别下标)
        box_class = tf.argmax(box_score, axis=-1)  # [16,16,5]
        # print(box_class.shape,"class")
        # 根据下标获取该类别概率(score最大的为类别概率)
        box_class_score = tf.reduce_max(box_score, axis=-1)  # [16,16,5]

        # 筛选信息
        # 判断范围,若概率过小不进行画框,返回布尔
        pred_mask = box_class_score > 0.8 # 获取筛选过后的坐标
        boxes = tf.boolean_mask(boxes, pred_mask)  # [16,16,5,4]->[4]
        # 获取筛选过后的概率
        scores = tf.boolean_mask(box_class_score, pred_mask)  # [16,16,5] -> [N]
        # 获取筛选过后的类别
        classes = tf.boolean_mask(box_class, pred_mask)  # [16,16，5]-> [N]
        # 这里一定要写*IMGSZ,否则无法画
        boxes = boxes * IMGSZ

        if all_boxes==[]:
            all_boxes=boxes
            all_scores=scores
            all_classes=classes
        else:
            all_boxes=tf.concat([all_boxes,boxes],axis=0)
            all_scores = tf.concat([all_scores, scores], axis=0)
            all_classes = tf.concat([all_classes, classes], axis=0)
    # 非极大值抑制,防止画过多框,就根据scores进行限制,优先画分最大的,最多画40个
    select_idx = tf.image.non_max_suppression(all_boxes, all_scores, 40, iou_threshold=0.3)
    boxes = tf.gather(all_boxes, select_idx)  # [4]
    scores = tf.gather(all_scores, select_idx)  # [N]
    classes = tf.gather(all_classes, select_idx)  # [N]

    # 读取图片
    # 总框数
    n_boxes = boxes.shape[0]
    img = img[..., ::-1]
    img = (np.array(img[0]) * 255).astype('uint8')
    for i in range(n_boxes):

        # 坐标信息
        x1, y1, x2, y2 = boxes[i]
        label = classes[i].numpy()

        # 设置颜色
        if label == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # 画矩形框

        cv2.rectangle(img, (x1.numpy(), y1.numpy()), (x2.numpy(), y2.numpy()), color=color, thickness=2)
        # 概率
        label_name = {}
        for n in range(num_work):
            label_name[n] = obj_name[n]
        label_name = {0: "watermelon", 1: "skin"}
        cv2.putText(img, f"{scores[i] * 100:.1f}%", (x1.numpy(), y1.numpy()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        # 添加
    return img.astype('uint8'), n_boxes, scores, classes

def visualize_img(img,model):

    img=visualize_rectangle(img,model, "cfg/watermelon.data")

    return img

def dataAnalysis(classes,scores):
    # 设置字体
    my_font = font_manager.FontProperties(fname="GUI/simsun.ttc",size=18)
    # 设置画板
    fig, ax = plt.subplots(figsize=(6, 3))
    # 默认数据
    recipe = ["0 健康",
              "0 炭疽病",
              "0 枯萎病",
              "0 虫害",
              "0 叶枯病",
              "0 白粉病"]
    data = np.array([float(x.split()[0]) for x in recipe])  # 1 , 0 ,
    ingredients = np.array([x.split()[-1] for x in recipe])  # 健康 炭疽病 枯萎病

    # 选择类别,并找出最大概率
    for i, cla in enumerate(classes):
        # if data[cla]<scores[i]:
        data[cla] += scores[i] * 100

    # 最可能的类型和概率
    max_score = np.max(data)/ np.sum(data)
    max_cla = np.argmax(data)

    # 排除概率为0的数据
    mask = (data != 0)
    data = data[mask]
    ingredients = ingredients[mask]

    def func(pct, allvals):
        absolute = int(round(pct / 100. * np.sum(allvals)))
        return "{:.1f}%".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"), startangle=90)

    ax.legend(wedges, ingredients,
              loc="center left",
              bbox_to_anchor=(0.9, 0, 0.5, 1), prop=my_font)

    plt.setp(autotexts, size=8, weight="bold")
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba()).astype('uint8')

    return img,max_cla,max_score

if __name__ == '__main__':
    data_cfg= "cfg/watermelon.data"
    model = yolo_Model()
    #model.build(input_shape=(None, 512, 512, 3))

    model.load_weights(r"weights/watermelon/watermelon")
    files = glob.glob('data/watermelon/val/image/*.jpg')
    for x in files:
        visualize_result(x, model,data_cfg)
    plt.show()



