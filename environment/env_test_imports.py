import numpy as np
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import torch

x = torch.rand(100)
x.shape

import cortex

import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pycocotools.coco import COCO

from PROJECT_DNFFA.ANALYSES import hello

hello()

