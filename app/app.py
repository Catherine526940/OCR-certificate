import json
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io

from ImageOCR import ImageOCR
from EasyOCR import EasyOCR

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


if __name__ == "__main__":
    app.run(debug=True)
