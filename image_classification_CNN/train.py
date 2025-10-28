import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleCNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(35,scale=(0.8,1.0)),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load datasets
train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
val_data = datasets.ImageFolder(root='dataset/val', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
print(f'Training classes: {train_data.classes}')

# Initialize model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
num_epochs = 50
train_losses = []
val_losses = []
metrics = {'train_acc': [],
           'val_acc': [],
           'train_prec': [],
           'val_prec': [],
           'train_rec': [],
           'val_rec': [],
           'train_f1': [],
           'val_f1': []}

best_val_loss=float('inf')
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_prec = precision_score(all_labels, all_preds, average='weighted')
    train_rec = recall_score(all_labels, all_preds, average='weighted')
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    train_losses.append(train_loss)
    metrics['train_acc'].append(train_acc)
    metrics['train_prec'].append(train_prec)
    metrics['train_rec'].append(train_rec)
    metrics['train_f1'].append(train_f1)


    model.eval()
    val_running_loss = 0.0
    val_labels=[]
    val_preds=[]

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            preds=torch.argmax(outputs,dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())


    val_loss = val_running_loss / len(val_loader)
    val_acc=accuracy_score(val_labels,val_preds)
    val_prec=precision_score(val_labels,val_preds,average='weighted')
    val_rec=recall_score(val_labels,val_preds,average='weighted')
    val_f1=f1_score(val_labels,val_preds,average='weighted')


    val_losses.append(val_loss)
    metrics['val_acc'].append(val_acc)
    metrics['val_prec'].append(val_prec)
    metrics['val_rec'].append(val_rec)
    metrics['val_f1'].append(val_f1)



    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
    print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}")

    if val_loss<best_val_loss:
        best_val_loss=val_loss
        best_model=model

# Save the model
current_time=time.time()
os.mkdir(str(current_time))
torch.save(model.state_dict(), f'{current_time}/last.pth')
torch.save(best_model.state_dict(),f'{current_time}/best.pth')


# Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('metrics.csv', index=False)


fig_dir=str(current_time)
# Plot the loss and metrics curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig(f'{fig_dir}/loss_curve.png')

plt.figure()
plt.plot(metrics['train_acc'], label='Train Accuracy')
plt.plot(metrics['val_acc'], label='Validation Accuracy')
plt.legend()
plt.savefig(f'{fig_dir}/accuracy_curve.png')

plt.figure()
plt.plot(metrics['train_prec'], label='Train Precision')
plt.plot(metrics['val_prec'], label='Validation Precision')
plt.legend()
plt.savefig(f'{fig_dir}/precision_curve.png')

plt.figure()
plt.plot(metrics['train_rec'], label='Train Recall')
plt.plot(metrics['val_rec'], label='Validation Recall')
plt.legend()
plt.savefig(f'{fig_dir}/recall_curve.png')

plt.figure()
plt.plot(metrics['train_f1'], label='Train F1-Score')
plt.plot(metrics['val_f1'], label='Validation F1-Score')
plt.legend()
plt.savefig(f'{fig_dir}/f1_score_curve.png')

