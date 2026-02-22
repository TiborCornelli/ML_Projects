import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class KeplerDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        # Redefine labels: 0 = no exoplanet, 1 = exoplanet
        self.y = torch.tensor(df.iloc[:, 0].values - 1, dtype=torch.float32)
        
        raw_flux = df.iloc[:, 1:].values
        mu = np.mean(raw_flux, axis=1, keepdims=True)
        std = np.std(raw_flux, axis=1, keepdims=True)
        self.x_raw = torch.tensor((raw_flux - mu) / std, dtype=torch.float32)
        
        fft_vals = np.abs(np.fft.fft(raw_flux, axis=1))
        fft_half = fft_vals[:, :raw_flux.shape[1]//2]
        fft_mean = np.mean(fft_half, axis=1, keepdims=True)
        fft_std = np.std(fft_half, axis=1, keepdims=True)
        fft_normalized = (fft_half - fft_mean) / (fft_std + 1e-8)
        self.x_fft = torch.tensor(fft_normalized, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_raw[idx].unsqueeze(0), self.x_fft[idx].unsqueeze(0), self.y[idx]

class DualStreamModel(nn.Module):
    def __init__(self, raw_len=3197, fft_len=1598):
        super().__init__()
        
        self.raw_stream = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        self.fft_stream = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_raw = torch.zeros(1, 1, raw_len)
            dummy_fft = torch.zeros(1, 1, fft_len)
            n_flat = self.raw_stream(dummy_raw).shape[1] + self.fft_stream(dummy_fft).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Linear(n_flat, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x_raw, x_fft):
        feat_raw = self.raw_stream(x_raw)
        feat_fft = self.fft_stream(x_fft)
        combined = torch.cat((feat_raw, feat_fft), dim=1)
        return self.classifier(combined).squeeze()

class RawOnlyModel(nn.Module):
    def __init__(self, raw_len=3197):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, raw_len)
            n_flat = self.feature_extractor(dummy_input).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Linear(n_flat, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x_raw, x_fft=None):
        features = self.feature_extractor(x_raw)
        return self.classifier(features).squeeze()

class FFTOnlyModel(nn.Module):
    def __init__(self, fft_len=1598):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, fft_len)
            n_flat = self.feature_extractor(dummy_input).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Linear(n_flat, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x_raw, x_fft):
        features = self.feature_extractor(x_fft)
        return self.classifier(features).squeeze()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        return torch.relu(out)

class LargeResNetModel(nn.Module):
    def __init__(self, raw_len=3197):
        super().__init__()
        
        self.prep = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 128)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x_raw, x_fft=None):
        x = self.prep(x_raw)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return self.classifier(x).squeeze()