# Load datasets

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

train_data = datasets.ImageFolder(root='~/image_classification_split/train', transform=transform)
val_data = datasets.ImageFolder(root='~/image_classification_split/val', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)