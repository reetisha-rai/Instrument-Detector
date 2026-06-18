"""
src/model.py — CNN-BiLSTM with Self-Attention for multi-label instrument detection.
Architecture is identical to what was trained in Colab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT, N_MELS, NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d((2, 1))
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.drop(self.pool(self.net(x)))


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.s = d ** 0.5

    def forward(self, x):
        a = F.softmax(torch.bmm(self.q(x), self.k(x).transpose(1, 2)) / self.s, dim=-1)
        return torch.bmm(a, self.v(x)) + x


class InstrumentDetector(nn.Module):
    def __init__(self):
        super().__init__()
        ch        = [1] + CNN_CHANNELS
        self.cnn  = nn.ModuleList([ConvBlock(ch[i], ch[i + 1]) for i in range(len(CNN_CHANNELS))])
        rm        = N_MELS // (2 ** len(CNN_CHANNELS))
        lin       = CNN_CHANNELS[-1] * rm
        self.lstm = nn.LSTM(lin, LSTM_HIDDEN, LSTM_LAYERS,
                            batch_first=True, bidirectional=True, dropout=DROPOUT)
        lo        = LSTM_HIDDEN * 2
        self.attn = Attention(lo)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(lo, 128), nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        for b in self.cnn:
            x = b(x)
        B, C, F, T = x.shape
        x, _ = self.lstm(x.permute(0, 3, 1, 2).reshape(B, T, C * F))
        return self.head(self.attn(x).mean(1))


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load a saved checkpoint. Returns (model, metadata_dict)."""
    model = InstrumentDetector().to(device)
    ckpt  = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt
