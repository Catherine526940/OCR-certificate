import math
import os.path

import numpy as np
import cv2
import pandas as pd

from ImageProcessing.PreprocessConfig import config


class ImageProcessor:
    def __init__(self):
        self.img_now_rotate = None

    def setImage(self, img, shape):
        img = cv2.resize(img, shape)
        img, angle = self._rotate(img)
        red_channel = img.copy()[:, :, 2]
        self.img_now_rotate = red_channel
        return red_channel, angle

    def _getRotateDegree(self, thetas):
        # 前提：倾斜角度不超过正负45度！
        res_all = []
        for value in thetas:
            res = value / np.pi * 180
            # 左偏
            if 45 <= res < 90:
                res -= 90
            elif 135 <= res < 180:
                res -= 180
            # 右偏
            elif 90 <= res < 135:
                res -= 90
            # print(res)
            res_all.append(res)
        return sum(res_all) / len(res_all)

    def _rotate(self, image):
        houghline_scale = 2
        canny = cv2.Canny(image, 50, 200, 3)
        lines = cv2.HoughLines(canny, 1, np.pi / 360, min(canny.shape) // houghline_scale)
        while lines is None and houghline_scale < 6:
            houghline_scale += 1
            lines = cv2.HoughLines(canny, 1, np.pi / 360, min(canny.shape) // houghline_scale)
        if lines is None:
            return image, 0
        values = []
        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            values.append(theta)
        angle = self._getRotateDegree(values)  # 角度
        M = cv2.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), angle, 1)
        result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), None, cv2.INTER_LINEAR,
                                cv2.BORDER_REPLICATE,
                                (255, 255, 255))
        return result, angle

    def doOtherProcess(self):
        '''
        对当前图像做其他处理。

        不在这里调整预处理的具体参数。

        :param image: np.array 需要预处理的图片，cv2读取
        '''
        if config.use_preprocess:
            image = self.img_now_rotate.copy()
            if config.use_filter:
                if config.filter_type.lower() == "gaussian":
                    image = cv2.GaussianBlur(image, (config.filter_kernel_size, config.filter_kernel_size), 0)
                elif config.filter_type.lower() == "mean":
                    image = cv2.medianBlur(image, config.filter_kernel_size)
            elif config.use_edge_sharpen:
                edge = cv2.Laplacian(image, -1, config.edge_sharpen_kernel)
                image = cv2.add(image, edge)
            if config.use_threshold:
                image = cv2.adaptiveThreshold(image, 255, config.threshold_method, cv2.THRESH_BINARY, config.threshold_kernel,
                                              config.threshold_C)
            if config.dilate_first:
                image = cv2.dilate(image, np.ones((config.dilate_kernel_size, config.dilate_kernel_size),
                                                  dtype=np.uint8), config.dilate_time)
                image = cv2.erode(image, np.ones((config.erode_kernel_size, config.erode_kernel_size),
                                                 dtype=np.uint8), config.erode_time)
            else:
                image = cv2.erode(image, np.ones((config.erode_kernel_size, config.erode_kernel_size),
                                                 dtype=np.uint8), config.erode_time)
                image = cv2.dilate(image, np.ones((config.dilate_kernel_size, config.dilate_kernel_size),
                                                  dtype=np.uint8), config.dilate_time)
            return image
        else:
            return self.img_now_rotate

    def changeConfig(self, data):
        '''
        在这里设置预处理的具体参数。

        :param config:json 配置的具体参数，见PreprocessConfig.py
        '''
        config.setConfig(data)

def _markItemsWithFixed(poses, img):
    for (index, x1, y1, x2, y2) in poses.itertuples():
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    showImg("", img)


def showImg(title, img):
    w = img.shape[0]
    h = img.shape[1]
    cv2.imshow(title, cv2.resize(img, (w // 4, h // 4)))
    cv2.waitKey(0)


if __name__ == "__main__":
    def readTemplateTxt(template_path):
        template = []
        h, w = 0, 0
        with open(template_path, "r", encoding="utf8") as f:
            lines = f.readlines()
        row = len(lines)
        for index, line in enumerate(lines):
            line = line.strip("\n").split(" ")
            if index != row - 1:
                template.append(line)
            else:
                h, w = map(int, line)
        return h, w, pd.DataFrame(template, columns=["name", "ltx", "lty", "rbx", "rby"])
    detector = ImageProcessor()
    # img_path = r"D:\DLDataset\gaoya\1.jpg"
    img_path = r"D:\DLDataset\cqzp\26.jpg"
    image = cv2.imread(img_path)
    # img_path = r"D:\ProgramCode\craft_rcnn_forch-master\craft_rcnn_forch\data\test_img\why.jpg"
    image_h, image_w, template = readTemplateTxt("产权证.txt")
    _, angle = detector.setImage(image, shape=(image_w, image_h))
    copy = detector.doOtherProcess()
    _markItemsWithFixed(template.iloc[1:], copy)
