import numpy as np
import torch
import torchvision
import ffcv
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import wandb 
from pycocotools.coco import COCO

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
import gc
import torch

x = torch.rand(100)

x.to('cuda:0')

del x 

torch.cuda.empty_cache()
gc.collect()

import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pycocotools.coco import COCO
import graphviz
import torchlens
import nibabel as nib
import seaborn as sns
import h5py
import sklearn
import wandb
import cortex

from jsputils import encoding, lesioning, losses, nnutils, nsdorg, paths, plotting, readout, selectivity, statsmodels, training, validation

print('finished successfully.')

