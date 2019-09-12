import numpy as np
import os.path
import glob
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
ENCODE_METHOD = 'utf-8'

class YOLO_Kmeans:

    def __init__(self, cluster_number, filepath):
        self.cluster_number = cluster_number
        self.filepath = filepath

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        """
        计算每个box与k个聚类中心的iou
        :param boxes:
        :param clusters:
        :return:
        """
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1] #n行1列
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n]) #repeate cluster_area n times
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix) #找出cluster_h_matrix中与box_h_matrix的
        inter_area = np.multiply(min_w_matrix, min_h_matrix) #并不是矩阵乘法只是相对应位置上的元素相乘。

        result = inter_area / (box_area + cluster_area - inter_area) # 计算IOU
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters) #distance
            current_nearest = np.argmin(distances, axis=1) #返回最小值的索引判断每一个样本距离哪一个cluster最近。
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0) #axis=0是按行，axis=1是按列

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def addShape(self, bndbox):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        width_trans = (xmax - xmin)
        height_trans = (ymax-ymin)
        points = [width_trans,height_trans]
        return points



    def txt2boxes(self):
        """
        解析xml文件获取gt的高和宽
        :return:
        """
        dataSet = []
        for file in os.listdir(self.filepath):
            filepath = os.path.join(self.filepath,file)
            parser = etree.XMLParser(encoding=ENCODE_METHOD)
            xmltree = ElementTree.parse(filepath, parser=parser).getroot()
            for object_iter in xmltree.findall('object'):
                bndbox = object_iter.find("bndbox")
                dataSet.append(self.addShape(bndbox))
        result = np.array(dataSet)
        print("initial w and h:",result)
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])] #排序类似于argsort()
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filepath = "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/"
    kmeans = YOLO_Kmeans(cluster_number, filepath)
    kmeans.txt2clusters()

# import sys
# from xml.etree import ElementTree
# from xml.etree.ElementTree import Element, SubElement
# from lxml import etree
# import numpy as np
# import os
# import sys
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import json
#
# XML_EXT = '.xml'
# ENCODE_METHOD = 'utf-8'
#
# #pascalVocReader readers the voc xml files parse it
# class PascalVocReader:
#     """
#     this class will be used to get transfered width and height from voc xml files
#     """
#     def __init__(self, filepath,width,height):
#         # shapes type:
#         # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
#         self.shapes = []
#         self.filepath = filepath
#         self.verified = False
#         self.width=width
#         self.height=height
#
#         try:
#             self.parseXML()
#         except:
#             pass
#
#     def getShapes(self):
#         return self.shapes
#
#     def addShape(self, bndbox, width,height):
#         xmin = int(bndbox.find('xmin').text)
#         ymin = int(bndbox.find('ymin').text)
#         xmax = int(bndbox.find('xmax').text)
#         ymax = int(bndbox.find('ymax').text)
#         width_trans = (xmax - xmin)/width*self.width
#         height_trans = (ymax-ymin)/height *self.height
#         points = [width_trans,height_trans]
#         self.shapes.append((points))
#
#     def parseXML(self):
#         assert self.filepath.endswith(XML_EXT), "Unsupport file format"
#         parser = etree.XMLParser(encoding=ENCODE_METHOD)
#         xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
#         pic_size = xmltree.find('size')
#         size = (int(pic_size.find('width').text),int(pic_size.find('height').text))
#         for object_iter in xmltree.findall('object'):
#             bndbox = object_iter.find("bndbox")
#             self.addShape(bndbox, *size)
#         return True
#
# class create_w_h_txt:
#     def __init__(self,vocxml_path,txt_path):
#         self.voc_path = vocxml_path
#         self.txt_path = txt_path
#     def _gether_w_h(self):
#         pass
#     def _write_to_txt(self):
#         pass
#     def process_file(self):
#         file_w = open(self.txt_path,'a')
#        # print (self.txt_path)
#         for file in os.listdir(self.voc_path):
#             file_path = os.path.join(self.voc_path, file)
#             xml_parse = PascalVocReader(file_path,304,304)
#             data = xml_parse.getShapes()
#             for w,h in data :
#                 txtstr = str(w)+' '+str(h)+'\n'
#                 #print (txtstr)
#                 file_w.write(txtstr)
#         file_w.close()
#
# class kMean_parse:
#     def __init__(self,path_txt):
#         self.path = path_txt
#         self.km = KMeans(n_clusters=5,init="k-means++",n_init=10,max_iter=3000000,tol=1e-3,random_state=0)
#         self._load_data()
#
#     def _load_data (self):
#         self.data = np.loadtxt(self.path)
#
#     def parse_data (self):
#         self.y_k = self.km.fit_predict(self.data)
#         print(self.km.cluster_centers_)
#         f = open('yolo_anchor.txt','w')
#         f.write(json.dumps(self.km.cluster_centers_.tolist()))
#         f.close()
#
#
#     def plot_data (self):
#         plt.scatter(self.data[self.y_k == 0, 0], self.data[self.y_k == 0, 1], s=50, c="orange", marker="o", label="cluster 1")
#         plt.scatter(self.data[self.y_k == 1, 0], self.data[self.y_k == 1, 1], s=50, c="green", marker="s", label="cluster 2")
#         plt.scatter(self.data[self.y_k == 2, 0], self.data[self.y_k == 2, 1], s=50, c="blue", marker="^", label="cluster 3")
#         plt.scatter(self.data[self.y_k == 3, 0], self.data[self.y_k == 3, 1], s=50, c="gray", marker="*",label="cluster 4")
#         plt.scatter(self.data[self.y_k == 4, 0], self.data[self.y_k == 4, 1], s=50, c="yellow", marker="d",label="cluster 5")
#        # draw the centers
#         plt.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], s=250, marker="*", c="red", label="cluster center")
#         plt.legend()
#         plt.grid()
#         plt.show()
#
#     def w1(self):
#         pass
#
#
# if __name__ == '__main__':
#      whtxt = create_w_h_txt("./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/","./data1.txt") #指定为voc标注路径，以及存放生成文件路径
#      whtxt.process_file()
#      kmean_parse = kMean_parse("./data1.txt")#路径和生成文件相同。
#      kmean_parse.parse_data()
#      kmean_parse.plot_data() #绘图部分只支持五个簇，要增加，需要自家改代码即可
#
#
