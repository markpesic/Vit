from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm

from VIT.vit import Vit
from VIT.utils import WarmupCosineSchedule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 512

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=tr.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=tr.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

image_size = (32, 32)
patch_size = (8, 8)
num_classes = 100
dim = 1024
depth = 8
hidden_dim = 2048

model = Vit(image_size, patch_size, num_classes, dim, depth, hidden_dim)

model.to(device)

optimizer = optim.Adam(model.parameters(), 3e-3, betas = (0.9, 0.999))
scheduler = CosineAnnealingWarmRestarts(optimizer, 10000, 1)
criterion = nn.CrossEntropyLoss()

def train_loop(model, optimizer, trainloader, criterion, scheduler, device):
    tk0 = tqdm(trainloader)
    train_loss = []

    for batch, y in tk0:
        batch = batch.to(device)
        y = y.to(device)

        logits = model(batch)
        loss = criterion(logits.view(-1, 100), y.view(-1))
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        del batch, y, logits
    return train_loss

epochs = 300

for epoch in range(epochs):
    train_loss = train_loop(model, optimizer, trainloader, criterion, scheduler, torch.device('cuda'))
    print(np.mean(train_loss))
