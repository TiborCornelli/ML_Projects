import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

class LFWSiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None, length=5000):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.length = length
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
                    self.person_to_images[entry] = imgs

        self.person_names = list(self.person_to_images.keys())

    def __getitem__(self, index):
        target = random.randint(0, 1)
        if target == 0:
            person = random.choice(self.person_names)
            img1_path, img2_path = random.sample(self.person_to_images[person], 2)
        else:
            p1, p2 = random.sample(self.person_names, 2)
            img1_path = random.choice(self.person_to_images[p1])
            img2_path = random.choice(self.person_to_images[p2])

        img1, img2 = Image.open(img1_path).convert("RGB"), Image.open(img2_path).convert("RGB")
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
            
        return img1, img2, torch.tensor([float(target)])

    def __len__(self):
        return self.length

def save_inference_examples(model, dataset, device, filename='predictions.png', num_examples=5):
    model.eval()
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 20))
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for i in range(num_examples):
            img1, img2, label = dataset[i]
            img1_d, img2_d = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
            
            o1, o2 = model(img1_d, img2_d)
            dist = nn.functional.pairwise_distance(o1, o2).item()
            
            img1_plot = inv_normalize(img1).permute(1, 2, 0).cpu().numpy()
            img2_plot = inv_normalize(img2).permute(1, 2, 0).cpu().numpy()
            
            axes[i, 0].imshow(np.clip(img1_plot, 0, 1))
            axes[i, 1].imshow(np.clip(img2_plot, 0, 1))
            
            match_label = "SAME" if label.item() == 0 else "DIFFERENT"
            axes[i, 0].set_title(f"Actual: {match_label}")
            axes[i, 1].set_title(f"Dist: {dist:.4f}")
            for ax in axes[i]: ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_lfw(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = LFWSiameseDataset(data_path, transform=train_transform, length=5000)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    loss_history = []
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            o1, o2 = model(img1, img2)
            loss = criterion(o1, o2, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'siamese_resnet18.pth')
    print("Model weights saved to siamese_resnet18.pth")

    plt.figure()
    plt.plot(loss_history)
    plt.title("Contrastive Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss_plot.png')
    plt.close()
    
    save_inference_examples(model, ds, device)
    print("Plots saved as loss_plot.png and predictions.png")

if __name__ == "__main__":
    PATH = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/"
    train_lfw(PATH)