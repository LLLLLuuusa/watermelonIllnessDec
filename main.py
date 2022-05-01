
import tensorflow as tf
import cv2
from PyQt5 import QtWidgets
import sys
import time
from PyQt5.QtGui import QPixmap, QImage, QPalette
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from inference import visualize_img,dataAnalysis
import numpy as np
from models.yolo import yolo_Model

tf.device(1)

from GUI.untitled import Ui_MainWindow
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)



        # 诊断图片
        self.diagnose_Button.clicked.connect(self.Diagnose)

        # 图片大小自适应i
        self.img_Label.setScaledContents(True)
        self.imAnlysis_Label.setScaledContents(True)
        self.img_Label.setStyleSheet("background-color: white ")

        # 打开资源
        self.openDir_Button.clicked.connect(self.Dir)

        # 图片地址
        self.imgPath = None

        self.img = None

        #个数
        self.n=None

        #概率
        self.scores = None

        #类别
        self.classes = None

        #获取所有摄像头
        self.showVidoeCapture()

        #判断摄像头是否打开
        self.VideoisOpen = False
    #展示所有摄像头,上限40个摄像头
    def showVidoeCapture(self):
        cnt = 0
        for device in range(0, 40):
            print(device)
            stream = cv2.VideoCapture(device)

            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                break
            self.videoCapturecomboBox.addItem(f"摄像头{device+1}")
            cnt = cnt + 1

    # 展示图片
    def ViewImg(self, Im):
        image_height, image_width, image_depth = Im.shape  # 获取图像的高，宽以及深度。\
        QIm = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data,image_width, image_height,QImage.Format_RGB888)
        self.img_Label.setPixmap(QPixmap.fromImage(QIm))# 将QImage显示在之前创建的QLabel控件中

    # 展示数据分析图
    def ImgAnlysis(self,ImAnlysis):
        image_height, image_width, image_depth = ImAnlysis.shape  # 获取图像的高，宽以及深度。\
        QIm = cv2.cvtColor(ImAnlysis, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height, QImage.Format_RGB888)
        self.imAnlysis_Label.setPixmap(QPixmap.fromImage(QIm))  # 将QImage显示在之前创建的QLabel控件中

    # 打开资源
    def Dir(self):
        if self.img_radioButton.isChecked():
            openfile_name = QFileDialog.getOpenFileName(self, '选择图片', '',
                                                        'img(*.jpg , *.png)')  # ('path', 'Excel files(*.jpg , *.png)')
            if openfile_name[0] != '':
                self.imgPath = openfile_name[0]
                self.img = cv2.imread(self.imgPath)  # 通过Opencv读入一张图片
                # 展示图片
                self.ViewImg(self.img)
        elif self.video_radioButton.isChecked():
            openfile_name = QFileDialog.getOpenFileName(self, '选择视频', '',
                                                        'mp4(*.mp4)')  # ('path', 'Excel files(*.mp4)')
            self.img=cv2.imread("GUI/mp4_demo.png")
            # 展示图片
            self.ViewImg(self.img)
        elif self.realTime_radioButton.isChecked():
            #打开摄像头
            cap = cv2.VideoCapture(self.videoCapturecomboBox.currentIndex())
            self.VideoisOpen = True
            while True:
                flage, self.img = cap.read()
                #若摄像头出现问题,直接报错
                if(flage!=True):
                    msg_box = QMessageBox(QMessageBox.Warning, '警告', '打开摄像头出现未知错误')
                    msg_box.exec_()
                    break
                self.ViewImg(self.img)
                cv2.waitKey(100)
                if not self.realTime_radioButton.isChecked() or not self.VideoisOpen:
                    break
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '没有选择模式')
            msg_box.exec_()
            return

    # 诊断
    def Diagnose(self):
        #判断选择的模式
        if self.img_radioButton.isChecked():
            #判断是否有图片
            if self.img is None:
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '没有上传图片')

                msg_box.exec_()
                return

            model = yolo_Model()
            # model = darknet_Model()
            model.build(input_shape=(None, 256, 256, 3))
            model.load_weights("weights/watermelon/watermelon")

            #调用yolo算法
            Im,n,scores,classes = visualize_img(self.img,model)#获取诊断的图像,个数和概率

            #传递数据
            self.ViewImg(Im)
            self.n=n
            self.scores=np.array(scores)
            self.number_label.setText(f"检查目标:{n}")
            self.classes=classes

            #调用数据分析算法
            imAnlysis,max_cla,max_score=dataAnalysis(self.classes,self.scores)
            self.ImgAnlysis(imAnlysis)

            #诊断方案
            if(max_cla==0):
                self.printf("<h3>当前状态:健康</h3>")
            elif(max_cla==1):
                self.printf("<h3>当前状态:炭疽病</h3><br>"
                        f"<h3>感染概率:{max_score:.1%}</h3><br>"
                        "发生炭疽病时，可摘除病叶、注意透光和通风，不要放置过密。药物防治用50%多菌灵可湿性粉剂500倍液浸种1小时，消灭种子表面的病菌，用水冲洗干净后播种")
        elif self.video_radioButton.isChecked():
            return
        elif self.realTime_radioButton.isChecked():
            self.VideoisOpen = False
            cap = cv2.VideoCapture(self.videoCapturecomboBox.currentIndex())
            model = yolo_Model()
            # model = darknet_Model()
            model.build(input_shape=(None, 256, 256, 3))
            model.load_weights("weights/watermelon/watermelon")
            while True:
                flage, self.img = cap.read()
                # 若摄像头出现问题,直接报错
                if(flage!=True):
                    msg_box = QMessageBox(QMessageBox.Warning, '警告', '打开摄像头出现未知错误')
                    msg_box.exec_()
                    break

                self.img=cv2.resize(self.img,(256,256))

                self.img, n, scores, classes = visualize_img(self.img,model)  # 获取诊断的图像,个数和概率
                self.ViewImg(self.img)
                cv2.waitKey(100)
                if not self.realTime_radioButton.isChecked():
                    break

            return
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '没有选择模式')
            msg_box.exec_()
            return
    # 诊断结果
    def printf(self, mes):
        self.diagnosticSchemeText.clear()
        self.diagnosticSchemeText.append(mes)  # 在指定的区域显示提示信息

            # self.cursot = self.DiagnosticSchemeText.textCursor()
            # self.DiagnosticSchemeText.moveCursor(self.cursot.End)
            # QtWidgets.QApplication.processEvents()

    #关闭窗口
    def closeEvent(self,event):
        sys.exit()


app = QtWidgets.QApplication(sys.argv)
window = mywindow()
window.show()
sys.exit(app.exec_())
