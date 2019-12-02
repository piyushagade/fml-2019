import sys, time, os
from itertools import product
from collections import OrderedDict

from modules import data as d
from modules import preprocess
from modules import cnn as cnn
from modules import mnist as mnist
from modules import k_cross_validation as kcv
from modules import globals as g
from modules import model as m
from skimage.color import rgb2gray
import skimage.filters as filt
from sklearn.model_selection import train_test_split
from sklearn import metrics


import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from torch.utils.data import DataLoader, TensorDataset
import pickle

from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.transforms as transforms
import torch.utils.data as data_utils
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")