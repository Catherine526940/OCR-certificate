from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections.abc
from PIL import Image, ImageFilter
import math
import random
import numpy as np
import cv2

EOS_TOKEN = 0  # special token for end of sentence
BLANK_TOKEN = 1  # special token for blank
UNKNOWN_TOKEN = 2
class ConvertBetweenStringAndLabel(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict_reverse = {}
        self.dict['EOS_TOKEN'] = EOS_TOKEN
        self.dict_reverse[EOS_TOKEN] = ""
        self.dict['BLANK_TOKEN'] = BLANK_TOKEN
        self.dict_reverse[BLANK_TOKEN] = ""
        # self.dict["UNKNOWN_TOKEN"] = UNKNOWN_TOKEN
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 2
            self.dict_reverse[i + 2] = item

    def encode(self, text):
        """
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.dict[item] if item in self.dict else 6113 for item in text]
            return torch.LongTensor(text)
        elif isinstance(text, collections.abc.Iterable):
            text = [self.encode(s) for s in text] # [text_num, words_encoded]
            text_length = [len(x) for x in text]
            max_length = max(text_length) # max length of sentences
            nb = len(text) # text_num
            targets = torch.zeros(nb, max_length + 1)
            for i in range(nb):
                targets[i][:len(text[i])] = text[i]
                targets[i][len(text[i])] = 0
            text = targets.transpose(0, 1).contiguous() # [words_encoded, text_num]
            text = text.long()
            return torch.LongTensor(text), torch.tensor(text_length)

    def decode(self, t):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = self.dict_reverse[t.item()]
        return texts if t != 1 else ""
    
    def decodeList(self, ls):
        chars = []
        lastchar = -1
        for char in ls:
            if char == EOS_TOKEN:
                break
            if char == lastchar:
                continue
            else:
                chars.append(self.decode(char))
                lastchar = char
        return "".join(chars)

class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v: Variable | torch.Tensor):
        count = None
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
