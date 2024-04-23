import math
import os.path

import numpy as np
import cv2
import pandas as pd

from ImageProcessing.PreprocessConfig import *


class ImageProcessor:
    def __init__(self):
        self.img_now_rotate = None

    def processImage(self, img, shape, relative=False, poses=None):
        """
        预处理的总方法。

        不在这里调整预处理的具体参数，这里只是传图片过来。

        :param img: np.array 需要预处理的图片，cv2读取
        :param shape: Iterable[2] 模板图像的宽高
        :param relative: bool 目前使用的是写死的绝对坐标。如果要改为以前的相对坐标，则改为True并填上poses。有bug!
        :param poses: list[][4] 见相对坐标模板，反正现在也不用了
        """
        img = cv2.resize(img, shape)
        img, angle = self._rotate(img)
        self.img_now_rotate = img
        red_channel = img.copy()[:, :, 2]
        if use_preprocess:
            red_channel = self.other_process(red_channel)
        if relative:
            tmpl_lt, tmpl_rb = self._findMinAndMaxPos(np.array(poses))
            # print(self.tmpl_lt, self.tmpl_rb)
            template_x_range = tmpl_rb[0] - tmpl_lt[0]
            template_y_range = tmpl_rb[1] - tmpl_lt[1]
            pos1, pos2 = self._FindTablePos(red_channel)
            now_x_range = pos2[0] - pos1[0]
            now_y_range = pos2[1] - pos1[1]
            x_scale = now_x_range / template_x_range
            y_scale = now_y_range / template_y_range
            poses = self._markItems(pos1, img, poses[1:], x_scale, y_scale)
            return poses, red_channel, angle
        else:
            # self._markItemsWithFixed(poses, img)
            return poses, red_channel, angle

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
        canny = cv2.Canny(image, 50, 200, 3)
        lines = cv2.HoughLines(canny, 1, np.pi / 360, min(canny.shape) // 4)
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

    def other_process(self, image):
        '''
        对图像做其他处理。

        不在这里调整预处理的具体参数。

        :param image: np.array 需要预处理的图片，cv2读取
        '''
        if use_threshold:
            image = cv2.adaptiveThreshold(image, 255, threshold_method, cv2.THRESH_BINARY, threshold_kernel, threshold_C)
        if use_filter:
            if filter_type.lower() == "gaussian":
                image = cv2.GaussianBlur(image, (filter_kernel_size, filter_kernel_size), 0)
            elif filter_type.lower() == "mean":
                image = cv2.medianBlur(image, filter_kernel_size)
        elif use_edge_sharpen:
            edge = cv2.Laplacian(image, -1, edge_sharpen_kernel)
            image = cv2.add(image, edge)
        if dilate_first:
            image = cv2.dilate(image, np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8), dilate_time)
            image = cv2.erode(image, np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8), erode_time)
        else:
            image = cv2.erode(image, np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8), erode_time)
            image = cv2.dilate(image, np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8), dilate_time)
        return image

    def changeConfig(self, config):
        '''
        在这里设置预处理的具体参数。

        :param config:json?dict? 配置的具体参数，见PreprocessConfig.py
        TODO: 这个功能尚未完成。
        '''
        setConfig(config)
        return self.other_process(self.img_now_rotate)


    def _FindTablePos(self, src_img):
        copy = src_img.copy()
        copy = cv2.GaussianBlur(copy, (3, 3), 0)
        # _, threshold_1 = cv2.threshold(copy, 160, 255, cv2.THRESH_OTSU)
        threshold = cv2.adaptiveThreshold(~copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -20)
        showImg("binary", threshold)
        # showImg("binary1", threshold_1)
        horizontal = threshold.copy()
        vertical = threshold.copy()
        scale = 20

        horizontalSize = int(horizontal.shape[1] / scale)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        verticalsize = int(vertical.shape[0] / scale)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        # showImg("vertical", vertical)
        mask = horizontal + vertical
        showImg("mask", mask)

        Net_img = cv2.bitwise_and(horizontal, vertical)
        netcopy = Net_img.copy()
        netcopy = cv2.dilate(netcopy, np.ones((5, 5), np.uint8))
        showImg("cross", netcopy)
        white_pixels = cv2.findNonZero(Net_img).reshape((-1, 2))
        sum_pos = white_pixels.sum(axis=1)
        min_pos = white_pixels[np.argmin(sum_pos)]
        max_pos = white_pixels[np.argmax(sum_pos)]
        return min_pos, max_pos

    def _markItems(self, min_pos, img, template_pos, scale_x, scale_y):
        min_x, min_y = min_pos[0], min_pos[1]
        poses = []
        for item in template_pos:
            x1 = min_x + math.floor(item[0] * scale_x)
            y1 = min_y + math.floor(item[1] * scale_y)
            x2 = x1 + math.ceil(item[2] * scale_x)
            y2 = y1 + math.ceil(item[3] * scale_y)
            poses.append((x1, y1, x2, y2))
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # showImg("", img)
        return poses

    def _findMinAndMaxPos(self, template_pos):
        return template_pos[0, :2], template_pos[0, 2:]

    def _markItemsWithFixed(self, poses, img):
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
    poses, copy, angle = detector.processImage(image, shape=(image_w, image_h))
    cv2.waitKey(0)
