from ImageProcessing.preprocess import ImageProcessor
from ournet.models.recognizer import RecognitionModel

import pandas as pd

class ImageOCR:
    def __init__(self, weights_path):
        self.changeMuban("产权证")
        self.image_processor = ImageProcessor()
        self.ocr = RecognitionModel(weights_path)
        self.currentmuban = "产权证"

    def changeMuban(self, name):
        self.image_h, self.image_w, self.template = self.readTemplateTxt(f"../ImageProcessing/{name}.txt")
        self.currentmuban = name

    def readTemplateTxt(self, template_path):
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

    def recognize(self, image):
        _, enhanced_image, angle = self.image_processor.processImage(image, self.template.iloc[:, 1:], (self.image_w, self.image_h))
        result = self.getResult(enhanced_image, image.shape, angle)
        return result

    def getResult(self, enhanced_image, raw_shape, angle):
        result_json = {}  # 整个json
        data_json = {}  # 第一个data
        data2_json = {}  # 第二个data
        kv_json = []  # 带坐标的
        for (i, (name, x1, y1, x2, y2)) in self.template.iterrows():
            result, conf = self.ocr.recognizeMultiline(enhanced_image, tuple(map(int, (x1, y1, x2, y2))))
            data2_json[name] = result
            item = self.generateKVInfo(name, 100, result, conf * 100, (x1, y1, x2, y2))
            kv_json.append(item)

        if self.currentmuban == "产权证":
            # 后处理补信息
            area = data2_json["面积"]
            house_area_index = area.find("房屋建筑")
            if house_area_index >= 0:
                house_area = data2_json["面积"][house_area_index + 6:].strip(" ")
                data2_json["房屋建筑面积"] = house_area
                kv_json.append(self.generateKVInfo("房屋建筑面积", 100, house_area, 60, None))
            else:
                data2_json["房屋建筑面积"] = ""
                kv_json.append(self.generateKVInfo("房屋建筑面积", 100, "", 60, None))

            chanquan = data2_json["证号"].replace(" ", "")
            hao_index = chanquan.find("号")
            data2_json["丘权号"] = chanquan[-8: -1] if hao_index >= 0 else chanquan[-7:]
            kv_json.append(self.generateKVInfo("丘权号", 100, chanquan[-8: -1], 80, None))

        # 组合json
        data_json["data"] = data2_json
        data_json["prism_keyValueInfo"] = kv_json
        data_json["orgWidth"] = enhanced_image.shape[1]
        data_json["orgHeight"] = enhanced_image.shape[0]
        data_json["width"] = raw_shape[1]
        data_json["height"] = raw_shape[0]
        data_json["angle"] = angle
        data_json["sid"] = "2b0855ab97ed5ce6f83b95df0c63b85c6381dda83c32c2ccc3c9617424e6f5de8f3a92ca"

        result_json["data"] = data_json
        result_json["code"] = "00001b001"
        result_json["msg"] = "操作成功"
        return result_json

    # 生成带坐标的json
    def generateKVInfo(self, key, keyProb, value, valueProb, pos=None):
        item = {}
        if pos is not None:
            item["valuePos"] = [{"x": pos[0], "y": pos[3]}, {"x": pos[0], "y": pos[1]}, {"x": pos[2], "y": pos[1]}, {"x": pos[2], "y": pos[3]}]
        item["keyProb"] = keyProb
        item["valueProb"] = valueProb
        item["value"] = value
        item["key"] = key
        return item
