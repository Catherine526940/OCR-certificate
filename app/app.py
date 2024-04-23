import json
import os

import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template, send_file
import base64
from PIL import Image
import io

from ImageOCR import ImageOCR
from EasyOCR import EasyOCR
from ImageProcessing.preprocess import ImageProcessor

app = Flask(__name__)
ocr = ImageOCR("../ournet/weights/model_6.pth")
# ocr = EasyOCR()

@app.route('/')
def index():
    return render_template('11.html')
# 设置新建模板页面的路由
@app.route('/new')
def new_template():
    # 在这里可以返回新建模板页面的 HTML 内容
    return render_template('new.html')
@app.route('/11')
def back_to_11():
    return render_template('11.html')
@app.route('/download', methods=['POST'])
def download_txt():
    # 从前端获取文档名称和内容
    document_name = request.form.get('documentName')
    tag_content = request.form.get('tagInput')
    image_width = request.form.get('imageWidth')
    image_height = request.form.get('imageHeight')

    # 指定文件保存路径
    file_path = '../ImageProcessing/' + document_name + '.txt'

    # 创建并写入到txt文件中
    with open(file_path, 'w', encoding="utf8") as file:
        file.write(tag_content.replace("\r", ""))
        file.write(f'{image_height} {image_width}')

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 返回文件作为响应并指定文件保存路径
        return send_file(file_path, as_attachment=True)
    else:
        return "Error: File not found"
@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    #image_file = request.data#z正常情况下传过来就是base64编码
    # print(request.data)

    image_file = json.loads(request.data)
    image_file = image_file["img"]
    image_file = image_file[image_file.index("base64,") + 7:]

    # 读取图像文件并解码为图像
    image_data = base64.b64decode(image_file)#base64解码
    image = Image.open(io.BytesIO(image_data))#用PIL加载图片
    #将PIL转为cv，因为后面我们切割（各种预处理）需要cv图片
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 图像处理与识别
    result = ocr.recognize(image)

    # 返回识别结果json
    return jsonify(result)

@app.route('/api/changemuban', methods=['POST'])
def change():
    name = json.loads(request.data)["name"]
    ocr.changeMuban(name)
    return jsonify(name)

@app.route('/api/processconfig')
def setConfig():
    config = json.loads(request.data)
    image_new = ocr.image_processor.changeConfig(config)
    return jsonify(image_new)

if __name__ == "__main__":
    app.run(debug=True)
