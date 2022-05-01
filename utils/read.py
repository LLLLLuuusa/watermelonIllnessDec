# ================================================================
#
#   Editor      : Pycharm
#   File name   : inference
#   Author      : LLLLLuuusa(HuangDaxu)
#   Created date: 2021-11-5 20:57
#   Email       : 1095663821@qq.com
#   QQ          : 1095663821
#   Description : 读取文件
#
#     (/≧▽≦)/ long mine the sun shine!!!
# ================================================================

import os
import xml.etree.ElementTree as ET
import numpy as np

# 解析xml文件
def parse_annotation(img_dir, ann_dir, labels):
    # img_dir指图片路径
    # ann_dir指选取框路径
    # labels指标签
    """
    <annotation>
	<folder>train</folder>
	<filename>X2-30-1.png</filename>
	<path /><source>
		<database>Unknown</database>
	</source>
	<size>
		<width>512</width>
		<height>512</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>sugarbeet</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>250</ymin>
			<xmax>53</xmax>
			<ymax>289</ymax>
		</bndbox>
	</object>
    """
    # 读取每一个ann文件
    imgs_info = []  # 把每一个ann文件的img_info以列表形式保存
    max_boxs = 0
    for ann in os.listdir(ann_dir):
        img_info = dict()  # [filename:,width:,height:,object:]
        img_info['object'] = []
        tree = ET.parse(os.path.join(ann_dir, ann))
        # 读取每一个主分支,并将分支对应的filename,width,hidth,object信息保存在img_info
        boxes_counter = 0
        for elem in tree.iter():
            if elem.tag == 'filename':
                img_info['filename'] = os.path.join(img_dir, elem.text)
            if elem.tag == 'size':
                for attr in elem.text:
                    if attr == 'width':
                        img_info['width'] = attr
                        # 这里本可以写一个变化,若图形不是512大小,就缩放为512,但是数据集都是512*512,就不使用此操作
                    if attr == 'height' == attr:
                        img_info['height'] = attr
                        # 这里本可以写一个变化,若图形不是512大小,就缩放为512,但是数据集都是512*512,就不使用此操作
            # object由[x,y,w,h,label]组成
            if elem.tag == 'object':
                object_info = [0, 0, 0, 0, 0]
                boxes_counter += 1
                for attr in elem.iter():
                    if attr.tag == 'name':
                        # label--0为不存在,1为sugarbeet,2为weed
                        label = labels.index(attr.text) + 1
                        object_info[4] = label
                    if attr.tag == 'bndbox':
                        for lab_size in attr.iter():
                            if lab_size.tag == 'xmin':
                                object_info[0] = lab_size.text
                            if lab_size.tag == 'ymin':
                                object_info[1] = lab_size.text
                            if lab_size.tag == 'xmax':
                                object_info[2] = lab_size.text
                            if lab_size.tag == 'ymax':
                                object_info[3] = lab_size.text
                img_info['object'].append(object_info)
        imgs_info.append(img_info)
        max_boxs = max(max_boxs, boxes_counter)

    # 将imgs_info中的filename和object分别保存至imgs和boxs
    imgs = []
    # 这里使用填充方案,每一个图片的bndbox数量都是不一样的,但是Np矩阵格式必须相同,所以就先找到最多的box为矩阵大小,其余填充0
    boxs = np.zeros([len(imgs_info), max_boxs, 5])
    for i, img_info in enumerate(imgs_info):
        img_boxes = np.array(img_info['object'])
        imgs.append(img_info['filename'])
        boxs[i, :img_boxes.shape[0]] = img_boxes

    # img:data\train\image\X.png   ---[b,]
    # boxs:[[  1. 170.  75. 321.   1.]  ---[b,40,5]
    #      [466. 385. 488. 403.   2.]
    #      [  0.   0.   0.   0.   0.]]
    return imgs, boxs

if __name__ == '__main__':
    obj_names = ('sugarbeet', 'weed')

    imgs, boxes = parse_annotation(r'..\data\val\image', r'..\data\val\annotation',obj_names)
    # print(imgs,boxes)