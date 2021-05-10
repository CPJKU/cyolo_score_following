import math
import librosa
import torch
import yaml

import numpy as np


def load_yaml(config_file):
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def load_wav(audio_path, sr):
    signal, _ = librosa.load(audio_path, sr=sr)

    return signal


def make_divisible(x, divisor):
    """taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py"""
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x):
    """taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py"""
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py"""
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def get_max_box(prediction):
    """
    Returns:
         most confident detection with shape: 1x4 (x, y, w, h)
    """

    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        _, idx = x[..., 4].max(-1)
        max_per_sample = x[idx][:4]
        output.append(max_per_sample)

    output = torch.stack(output)

    assert output.shape[0] == prediction.shape[0]
    return output


class AverageMeter(object):
    """
    Computes and stores the average and current value
    taken from https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
