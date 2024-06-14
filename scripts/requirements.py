import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PolynomialFeatures,
    Normalizer,
    StandardScaler,
    MinMaxScaler,
    StandardScaler,
)
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import time
import itertools
import json

import optuna
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler

type = "MLP_skip"

norm = "standardiztion"  ## define type of normalization
