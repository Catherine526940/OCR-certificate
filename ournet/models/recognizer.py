import numpy as np
import torch

import cv2

from ournet.src import utils1 as utils, dataset
from ournet.crnn import crnn
from ImageProcessing.cutintorows import detect_rows_full

class RecognitionModel:
    def __init__(self, weights_path):
        # load alphabet
        with open('../data/new_dict.txt', encoding="UTF-8") as f:
            data = f.readlines()
            alphabet = [x.rstrip() for x in data]
            self.alphabet = ''.join(alphabet)

        # define convert between string and label index
        self.converter = utils.ConvertBetweenStringAndLabel(self.alphabet)

        # len(alphabet) + SOS_TOKEN + EOS_TOKEN
        num_classes = len(self.alphabet) + 2

        img_width = 280
        img_height = 32
        self.transformer = dataset.ResizeNormalize(img_width=img_width, img_height=img_height, strict=False)
        self.model = crnn.CRNN(3, img_height, img_width, num_classes)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            map_location = 'cuda'
        else:
            map_location = 'cpu'

        self.model.load_state_dict(torch.load(weights_path, map_location=map_location))
        print(f'loading pretrained model from {weights_path}.')

        self.model.eval()

    def recognize(self, img):
        _img = self.transformer(img)
        _img = _img.view(1, *_img.size())
        _detect_img = torch.autograd.Variable(_img)
        if torch.cuda.is_available():
            _detect_img = _detect_img.to("cuda")
        output = self.model(_detect_img)
        topv, topi = output.detach().topk(1)  # [length, batch, 1]
        ni = topi.squeeze(2).transpose(0, 1)  # [batch, length]
        # 耗时较长
        decoded_words = list(map(self.converter.decodeList, ni))

        topv = topv.cpu().detach().numpy().reshape(-1)
        conf = np.prod(topv)**(2.0/np.sqrt(len(topv)))
        return ''.join(decoded_words), conf

    def recognizeMultiline(self, img, box):
        x1, y1, x2, y2 = box
        split_image = img[y1:y2, x1 + 5:x2 - 5]
        recog_text = []
        conf_result = 0
        is_multi, rows = detect_rows_full(split_image)
        text_count = len(rows)
        recog_image = img[y1:y2, x1 + 3:x2 - 3]
        recog_image = cv2.cvtColor(recog_image, cv2.COLOR_GRAY2RGB)
        w, h = img.shape[0], img.shape[1]
        for top, bottom in rows:
            if bottom - top < 10:
                continue
            top = top - 3 if top >= 3 else top
            bottom = bottom + 3 if bottom <= h - 3 else bottom
            sub_img = recog_image[top:bottom, :, :]
            # cv2.imshow("", sub_img)
            # cv2.waitKey(0)
            text, conf = self.recognize(sub_img)
            recog_text.append(text)
            conf_result += conf
        if text_count <= 0:
            return "", 1.0
        else:
            return "".join(recog_text), conf_result / text_count


if __name__ == '__main__':
    recognizer = RecognitionModel("../weights/model_6.pth")
    img = cv2.imread(r"D:\3.jpg", cv2.COLOR_RGB2GRAY)
    print(recognizer.recognize(img))
    cv2.imshow("img" ,img)
    cv2.waitKey(0)

