import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # If running on a headless server
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
from collections import defaultdict

# Transformers / HF
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

FUTURE_WORK_DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(FUTURE_WORK_DEMOS_DIR)
INFERENCE_EXPERIMENTS_DIR = os.path.join(REPO_DIR, "inference_experiments")
HEMAL_EXPERIMENTS_DIR = os.path.join(REPO_DIR, "hemal_experiments")
sys.path.append(INFERENCE_EXPERIMENTS_DIR)

from inference_experiment_utils import *

