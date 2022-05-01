# ================================================================
#
#   Editor      : Pycharm
#   File name   : groud_dataset
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-4 16:58
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 数据读取和预处理
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

from utils.preprocess import preprocess
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from utils.read import parse_annotation
import random
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import imgaug as ia
from    imgaug import augmenters as iaa

# 创建db文件
class Dataloader:
    def __init__(self,img_dir, ann_dir, bath, obj_names,ANCHORS,GRIDSZ,IMGSZ):
        #读取基本参数
        self.img_dir=img_dir
        self.ann_dir=ann_dir
        self.bath=bath
        self.obj_names=obj_names
        self.ANCHORS=ANCHORS
        self.GRIDSZ=GRIDSZ
        self.IMGSZ=IMGSZ

        self.aug=True

    def __call__(self):
        #将文件多线程处理为db文件
        db=self.get_dataset()
        db=self.augmentation_generator(db)
        #将db文件预处理
        gen=self.ground_truth_generator(db)

        return gen

    def __len__(self):
        return len(os.listdir(self.ann_dir))

    def get_dataset(self):
        # img_dir指图片路径
        # ann_dir指选取框路径
        # baths指线程
        imgs, boxs = parse_annotation(self.img_dir, self.ann_dir, self.obj_names)
        db = tf.data.Dataset.from_tensor_slices((imgs, boxs))
        db = db.shuffle(1000).map(preprocess).batch(self.bath).repeat()

        return db

    def augmentation_generator(self,yolo_dataset):
        '''
        Augmented batch generator from a yolo dataset

        Parameters
        ----------
        - YOLO dataset

        Returns
        -------
        - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            batch : tupple(images, annotations)
            batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        '''
        for batch in yolo_dataset:
            # conversion tensor->numpy
            img = batch[0].numpy()
            boxes = batch[1].numpy()
            # conversion bbox numpy->ia object
            ia_boxes = []
            for i in range(img.shape[0]):
                ia_bbs = [ia.BoundingBox(x1=bb[0],
                                         y1=bb[1],
                                         x2=bb[2],
                                         y2=bb[3]) for bb in boxes[i]
                          if (bb[0] + bb[1] + bb[2] + bb[3] > 0)]
                ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(256, 256)))
            # data augmentation
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),        #随机上下翻转
                iaa.Flipud(0.5),        #随机垂直翻转
                iaa.Multiply((0.4, 1.6)),  # 改变亮度
                #iaa.Crop(px=(0, 16)),  #缩进16个像素
                #iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
                #iaa.Affine(scale=(0.7, 1.30),translate_px={"x": (-100,100), "y": (-100,100)},rotate=(15,60)) #仿射变换
                iaa.Affine(scale=(0.7, 1.30), rotate=(-60, 60))# 仿射变换
            ])
            # seq = iaa.Sequential([])
            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images(img)
            img_aug = np.clip(img_aug, 0, 1)#防止数据增强后不符合范围在0~1
            boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
            # conversion ia object -> bbox numpy
            for i in range(img.shape[0]):
                boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
                for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                    boxes[i, j, 0] = bb.x1
                    boxes[i, j, 1] = bb.y1
                    boxes[i, j, 2] = bb.x2
                    boxes[i, j, 3] = bb.y2
            # conversion numpy->tensor
            batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
            # batch = (img_aug, boxes)
            yield batch


    # 预处理GTBOX,返回是否有最合适的候选框,候选框信息,新bndbox信息
    def process_true_boxes(self,gt_boxes):
        # 将bondbox数据[x1,y1,x2,y2]改为[x,y,w,h]
        gt_boxes_grid = np.zeros([3, gt_boxes.shape[0], gt_boxes.shape[1]])
        gt_boxes = gt_boxes.numpy()

        # 判断那个位置有那种合适的anchors(候选框)
        detector_masks = [np.zeros([self.GRIDSZ[0], self.GRIDSZ[0], 3, 1]), np.zeros([self.GRIDSZ[1], self.GRIDSZ[1], 3, 1]),
                          np.zeros([self.GRIDSZ[2], self.GRIDSZ[2], 3, 1])]
        # 候选框的x,y,w,h,l
        matching_gt_boxs = [np.zeros([self.GRIDSZ[0], self.GRIDSZ[0], 3, 5]), np.zeros([self.GRIDSZ[1], self.GRIDSZ[1], 3, 5]),
                            np.zeros([self.GRIDSZ[2], self.GRIDSZ[2], 3, 5])]

        for i, box in enumerate(gt_boxes):
            last_best_iou = 0
            for a in range(3):
                scale = self.IMGSZ // self.GRIDSZ[a]  # sacle=32
                anchors = np.array(self.ANCHORS[a]).reshape((3, 2))
                # # 判断那个位置有那种合适的anchors(候选框)
                # detector_mask = np.zeros([GRIDSZ[a], GRIDSZ[a], 3, 1])
                # # 候选框的x,y,w,h,l
                # matching_gt_box = np.zeros([GRIDSZ[a], GRIDSZ[a], 3, 5])

                x = (box[0] + box[2]) / 2 / scale
                y = (box[1] + box[3]) / 2 / scale
                w = (box[2] - box[0]) / scale
                h = (box[3] - box[1]) / scale

                gt_boxes_grid[a] = np.array([x, y, w, h, box[4]])

                if w * h > 0:
                    best_anchor = 0
                    best_iou = 0
                    for j in range(3):
                        # 相交部分
                        interct = np.minimum(w, anchors[j, 0]) * np.minimum(h, anchors[j, 1])
                        # 不相交部分
                        union = w * h + (anchors[j, 0] * anchors[j, 1]) - interct
                        # 交并比,iou最大则最合适
                        iou = interct / union
                        # 找到最合适候选框,并确认是否在本anchors中
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = j
                    if best_iou > 0 and best_iou >= last_best_iou:
                        # if last_best_iou!=0:
                        #     detector_masks[a-1]= detector_masks[a-1]*0
                        #     matching_gt_boxs[a-1] = matching_gt_boxs[a-1]*0

                        x_coord = np.floor(x).astype(np.int32)
                        y_coord = np.floor(y).astype(np.int32)
                        detector_masks[a][y_coord, x_coord, best_anchor] = 1
                        matching_gt_boxs[a][y_coord, x_coord, best_anchor] = np.array([x, y, w, h, box[4]])

                        last_best_iou = best_iou

        # mask: (3,)(8, 8, 3, 1)(16, 16, 3, 1)(32, 32, 3, 1)
        # gt_boxs: (3,)(8, 8, 3, 5)(16, 16, 3, 5)(32, 32, 3, 5)
        # boxes_grid: (3, 6, 5)(6, 5)(6, 5)(6, 5)
        return detector_masks, matching_gt_boxs, gt_boxes_grid

    # 对db文件进行预处理修改
    def ground_truth_generator(self,db):
        # db文件
        # ANCHORS [3,6]
        # ANCHORS 9
        # IMGSZ 256
        # GRIDS [32,16,8]
        for imgs, boxes in db:
            # imgs:[5,512,512,3]
            # boxes:[5,40,5]
            draw_img(imgs,boxes)

            bath_detector_mask = []
            bath_matching_gt_box = []
            bath_matching_classes_oh = []
            bath_gt_boxes_grid = []
            bath = imgs.shape[0]  # b=5

            for b in range(bath):
                detector_mask, matching_gt_box, gt_boxes_grid = \
                    self.process_true_boxes(boxes[b])

                matching_classes_ohs = []
                for a in range(3):
                    # 类别[b,16,16,5,3]
                    matching_classes = tf.cast(matching_gt_box[a][..., 4], tf.int32)
                    matching_classes_oh = tf.one_hot(matching_classes, depth=7)
                    # 由于类别只有两个,所以提取时[b,16,16,5,2]
                    matching_classes_ohs.append(tf.cast(matching_classes_oh[..., 1:], dtype=tf.float32))

                bath_detector_mask.append(detector_mask)
                bath_matching_gt_box.append(matching_gt_box)
                bath_matching_classes_oh.append(matching_classes_ohs)
                bath_gt_boxes_grid.append(gt_boxes_grid)

            # imgs(1, 256, 256, 3)
            # bath_detector_mask(1, 3)(16, 16, 3, 1)(16, 16, 3, 1)(32, 32, 3, 1)
            # bath_matching_gt_box(1, 3)(16, 16, 3, 5)(16, 16, 3, 5)(16, 16, 3, 5)
            # bath_matching_classes_oh(1, 3)(16, 16, 3, 6)(16, 16, 3, 6)(16, 16, 3, 6)
            # bath_gt_boxes_grid(1, 3, 6, 5)
            yield imgs, bath_detector_mask, bath_matching_gt_box, bath_matching_classes_oh, bath_gt_boxes_grid

# 可视化预处理后的db文件
def db_aug_visualize(gen):
    imgs, bath_detector_mask, bath_matching_gt_box, bath_matching_classes_oh, bath_gt_boxes_grid = \
        next(gen)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(5, 10))
    # # img:[b,512,512,3]
    # # bath_detector_mast:[b,16,16,5,1]
    ax1.imshow(imgs[0])
    mask=bath_detector_mask[0]
    mask1 = mask[0]
    mask2 = mask[1]
    mask3 = mask[2]
    mask1 = tf.reduce_sum(mask1, axis=2)
    mask2 = tf.reduce_sum(mask2, axis=2)
    mask3 = tf.reduce_sum(mask3, axis=2)

    ax2.imshow(mask1[..., 0])
    ax3.imshow(mask2[..., 0])
    ax4.imshow(mask3[..., 0])
    plt.show()

def draw_img(img,boxes):
    img = (np.array(img[0]) * 255).astype('uint8')
    boxes=np.array(boxes)
    for box in boxes[0]:
        # 坐标信息
        x1, y1, x2, y2,l = box
        # 设置颜色
        if l == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # 画矩形框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
    print(np.shape(img))
    print(type(img))
    cv2.imshow("demo", img)



        # 添加

if __name__ == '__main__':
    obj_names = ('normal','anthracnose','blight','insect_attack','leaf_blight','powdery_mildew')
    # ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    ANCHORS = [[0.19417476, 0.25242718,0.31067961, 0.58252427,0.6407767, 0.44660194],[1.16504854, 2.36893204,2.40776699, 1.74757282,2.29126214, 4.62135922],[9.00970874, 6.99029126,12.11650485, 15.37864078,28.97087379, 25.32038835]]
    IMGSZ = 256
    GRIDSZ = [8, 16, 32]
    dataloader = Dataloader(r'..\data\watermelon\test\image', r'..\data\watermelon\test\annotation', 5, obj_names,
                            ANCHORS, GRIDSZ, IMGSZ)
    gen = dataloader()
    db_aug_visualize(gen)
