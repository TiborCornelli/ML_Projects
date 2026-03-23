import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


class LFWSiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.person_to_images = {}

        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(full_path):
                imgs = [
                    os.path.join(full_path, f)
                    for f in os.listdir(full_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if len(imgs) >= 2:
                    self.person_to_images[full_path] = imgs

        self.person_folders = list(self.person_to_images.keys())
        print(f"Found {len(self.person_folders)} people with 2+ images.")

    def __getitem__(self, index):
        target = random.randint(0, 1)
        if target == 0:
            person = random.choice(self.person_folders)
            img1_path, img2_path = random.sample(self.person_to_images[person], 2)
        else:
            p1, p2 = random.sample(self.person_folders, 2)
            img1_path = random.choice(self.person_to_images[p1])
            img2_path = random.choice(self.person_to_images[p2])

        img1, img2 = Image.open(img1_path).convert("RGB"), Image.open(
            img2_path
        ).convert("RGB")
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, torch.tensor([float(target)])

    def __len__(self):
        return 1000


def train_lfw(data_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ds = LFWSiameseDataset(data_path, transform=t)
    if not ds.person_folders:
        print(f"Error: No person folders found in {data_path}")
        return

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            o1, o2 = model(img1, img2)
            loss = criterion(o1, o2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(script_dir, "archive", "lfw-deepfunneled", "lfw-deepfunneled")

    print(f"Targeting: {PATH}")
    train_lfw(PATH)
