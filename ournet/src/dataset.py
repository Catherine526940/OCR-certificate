import torch
import torch.utils.data
import torchvision

import random

from PIL import Image
import cv2
import numpy as np


class TextLineDataset(torch.utils.data.Dataset):

    def __init__(self, text_line_file=None, transform=None, target_transform=None):
        self.text_line_file = text_line_file
        with open(text_line_file, encoding="UTF-8") as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split()
        img_path = line_splits[0]
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = line_splits[1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class ResizeNormalize(object):

    def __init__(self, img_width, img_height, strict):
        self.img_width = img_width
        self.img_height = img_height
        self.toTensor = torchvision.transforms.ToTensor()
        self.strict_resizing = strict

    '''
    strict: 是否严格按照指定尺寸。默认为True。
            为False时，严格按照指定高度缩放，而宽度至少为self.width。（detect时需要）
    '''
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        height = self.img_height
        width = int(w * height / h)
        if self.strict_resizing:
            if width >= self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))
            else:
                img = cv2.resize(img, (width, height))
                img_pad = np.zeros((self.img_height, self.img_width, c), dtype=img.dtype)
                img_pad[:height, :width, :] = img
                img = img_pad
        else:
            img = cv2.resize(img, (width, height))
            if width < self.img_width:
                img_pad = np.zeros((self.img_height, self.img_width, c), dtype=img.dtype)
                img_pad[:height, :width, :] = img
                img = img_pad
        img = Image.fromarray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, batch_size):
        super(RandomSequentialSampler, self).__init__()
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batches = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batches):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, img_height=32, img_width=100):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = ResizeNormalize(img_width=self.img_width, img_height=self.img_height, strict=True)

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
