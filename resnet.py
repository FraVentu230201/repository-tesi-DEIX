import os
import csv
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MPIICsvDataset(Dataset):

    def __init__(self, csv_file, images_dir, is_train, transform, num_samples):
        super().__init__()
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                img_name = row[0]
                train_flag = int(row[1])
                if is_train and train_flag == 1:
                    self.samples.append(row)
                elif not is_train and train_flag == 0:
                    self.samples.append(row)

        if num_samples is not None and num_samples < len(self.samples):
            self.samples = random.sample(self.samples, num_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        row = self.samples[idx]
        img_name = row[0]
        coords_str = row[2:]
        coords = list(map(float, coords_str))
        coords = np.array(coords, dtype=np.float32)

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        img_width, img_height = image.size

        coords[::2] = coords[::2] / img_width
        coords[1::2] = coords[1::2] / img_height

        if self.transform:
            image = self.transform(image)

        coords_t = torch.from_numpy(coords)

        return image, coords_t, img_name


class ResNetPose(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_joints * 2)
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0

    for images, coords_gt, _ in loader:
        images = images.to(device)
        coords_gt = coords_gt.to(device)

        optimizer.zero_grad()
        coords_pred = model(images)
        loss = criterion(coords_pred, coords_gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, device):
    model.eval()
    funzione_perdita = nn.MSELoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, coords_gt, _ in loader:
            images = images.to(device)
            coords_gt = coords_gt.to(device)
            coords_pred = model(images)
            loss = funzione_perdita(coords_pred, coords_gt)
            running_loss += loss.item() 

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def allenamento():
    csv_file = "mpii_annotations.csv"
    images_dir = "images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    Dataset_training = MPIICsvDataset(csv_file, images_dir, True, train_transform,100)
    Dataset_validation = MPIICsvDataset(csv_file, images_dir, False, train_transform,100)

    train_loader = DataLoader(Dataset_training, batch_size=16, shuffle=True)
    val_loader = DataLoader(Dataset_validation, batch_size=16, shuffle=False)

    print(f"Train samples: {len(Dataset_training)}")
    print(f"Val   samples: {len(Dataset_validation)}")

    model = ResNetPose(num_joints=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "pose_regressor_mpii.pth")

    predictions_dict = {}

    with torch.no_grad():
        for images, coords_gt, img_names in train_loader:
            images = images.to(device)
            coords_gt = coords_gt.to(device)
            outputs = model(images)

            for i, name in enumerate(img_names):
                img_path = os.path.join(images_dir, name)
                image = Image.open(img_path)
                img_width, img_height = image.size

                pred_coords = outputs[i].detach().cpu().numpy().reshape(16, 2)
                pred_coords[:, 0] *= img_width
                pred_coords[:, 1] *= img_height

                real_coords = coords_gt[i].detach().cpu().numpy().reshape(16, 2)
                real_coords[:, 0] *= img_width
                real_coords[:, 1] *= img_height

                combined_array = np.stack((pred_coords, real_coords), axis=1)
                predictions_dict[name] = combined_array

    np.save("predictions_saved.npy", predictions_dict)
    return predictions_dict, train_loader


previsione_rete_neurale, train_loader = allenamento()
