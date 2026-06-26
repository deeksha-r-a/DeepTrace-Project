#README:
# this file contains all imports required for all the given 8 tasks

from datasets import load_dataset
import os
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
