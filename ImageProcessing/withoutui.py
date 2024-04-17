import json

import cv2

from preprocess import ImageProcessor
from app.start_his_own import OCR


class ImageOCR:
    def __init__(self):
        self.titles = []
        self.image_processor = ImageProcessor()
        self.ocr = OCR()
        with open("./title.txt", "r", encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            self.titles.append(line.strip("\n"))

    def recognize(self, image):
        _, bbox_pos, enhanced_image = self.image_processor.processImage(image)
        result = self.getResult(enhanced_image, bbox_pos)

    def getResult(self, enhanced_image, bbox_pos):
        result_json = []
        for index, (x1, y1, x2, y2) in enumerate(bbox_pos):
            result_text_tmp = {}
            image = enhanced_image[y1:y2, x1:x2]
            result, conf = self.ocr.recognize(image)
            # result_text_tmp["坐标"] = f"({x1},{y1}) - ({x2},{y2})"
            result_text_tmp[self.titles[index]] = result
            # result_text_tmp["置信度"] = conf
            result_json.append(result_text_tmp)

            with open('./output_file.json', 'a', encoding='utf-8') as f:
                json.dump(result_text_tmp, f, ensure_ascii=False)
                f.write("\n")
        return result_json


if __name__ == "__main__":
    ocr = ImageOCR()
    image = cv2.imread(r"D:\Users\95159\Desktop\1.jpg")
    ocr.recognize(image)

