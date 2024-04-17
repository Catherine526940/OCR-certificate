import os

import cv2

from easyocr.easyocr.easyocr import Reader

class OCR:
    def __init__(self):
        self.reader = Reader(['ch_sim', 'en'])

    def recognize(self, img):
        recog_text = []
        conf = 0
        ocr_result = self.reader.readtext(img)
        text_count = len(ocr_result)
        for detection in ocr_result:
            recog_text.append(detection[1])
            conf += detection[2]
        if text_count <= 0:
            return "", 1.0
        else:
            return "".join(recog_text), conf / text_count


def recognize_text_in_folder(folder_path, output_file):
    # 创建 EasyOCR 读取器对象
    reader = Reader(['ch_sim', 'en'])


    # 图片文件名与描述的对应关系字典
    file_descriptions = {
        '1.jpg': '权利人',
        '2.jpg': '共有情况',
        '3.jpg': '坐落',
        '4.jpg': '不动产单元号',
        '5.jpg': '权利类型',
        '6.jpg': '权利性质',
        '7.jpg': '用途',
        '8.jpg': '面积',
        '9.jpg': '使用期限',
        '91.jpg': '权力其他状况',
        '11.jpg': '附记',
        # 添加更多的图片文件名和对应描述
    }

    # 变量用于存储上一张图片文件名及其识别结果
    prev_filename = None
    prev_text = None

    # 遍历文件夹中的所有图片文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 获取当前图片文件的描述
                description = file_descriptions.get(filename, '未提供描述')

                # 写入图片文件名和对应的描述
                # f.write(f'File: {filename}\n')
                # f.write(f'Description: {description}\n')

                # 识别图片中的文本
                image_path = os.path.join(folder_path, filename)
                result = reader.readtext(image_path)

                # 将识别结果写入文件
                for detection in result:
                    coordinates = detection[0]
                    left_top = coordinates[0]
                    right_top = coordinates[1]
                    right_bottom = coordinates[2]
                    left_bottom = coordinates[3]
                    text = detection[1]
                    confidence = detection[2]

                    # 判断是否为同一张图片
                    if filename != prev_filename:
                        # 写入新的识别结果
                        f.write(f'x1,y1: {left_top}\n'
                        f'x2,y2: {right_top}\n'
                        f'x3,y3: {right_bottom}\n'
                        f'x4,y4: {left_bottom}\n'
                        f' {description}: {text},\n')
                       # f.write(f'Text: {text}\n')
                        prev_filename = filename
                        prev_text = text
                    elif text != prev_text:
                        # 避免重复输出相同的文字内容
                        f.write(f' {text}\n')
                        prev_text = text

                # 添加分隔符
                f.write('-' * 50 + '\n')

if __name__ == "__main__":
    # 调用函数来识别文件夹中的图片并保存结果到文件
    folder_path = './'  # 修改为你的文件夹路径
    output_file = './output_file.txt'  # 修改为你的输出文件路径
    recognize_text_in_folder(folder_path, output_file)