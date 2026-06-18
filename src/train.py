"""
src/train.py — training pipeline.

Usage:
    python src/train.py --dataset both --epochs 50
    python src/train.py --dataset irmas --epochs 20
    python src/train.py --dataset openmic --epochs 50
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR, CHECKPOINT_PATH,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOP,
    CONFIDENCE_THRESHOLD,
)
from src.model import InstrumentDetector
from src.dataset import IRMASDataset, OpenMICDataset, CombinedDataset


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy — down-weights easy negatives."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.g   = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        l = self.bce(logits, targets)
        return ((1 - torch.exp(-l)) ** self.g * l).mean()


# ── Train / eval helpers ──────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit, scaler, device):
    model.train()
    total = 0.0
    for specs, labels in tqdm(loader, desc='  train', leave=False):
        specs, labels = specs.to(device), labels.to(device)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                loss = crit(model(specs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss = crit(model(specs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, crit, device, thr=CONFIDENCE_THRESHOLD):
    model.eval()
    total  = 0.0
    ps, ls = [], []
    for specs, labels in tqdm(loader, desc='  val  ', leave=False):
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        total += crit(logits, labels).item()
        ps.append((torch.sigmoid(logits) > thr).float().cpu())
        ls.append(labels.cpu())
    p, l = torch.cat(ps), torch.cat(ls)
    tp = (p * l).sum().item()
    fp = (p * (1 - l)).sum().item()
    fn = ((1 - p) * l).sum().item()
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    f1 = 2 * pr * rc / (pr + rc + 1e-8)
    return {'loss': total / len(loader), 'precision': pr, 'recall': rc, 'f1': f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['irmas', 'openmic', 'both'], default='both')
    parser.add_argument('--epochs',  type=int,   default=EPOCHS)
    parser.add_argument('--batch',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--lr',      type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
    print(f'Device: {device}')

    DatasetClass = {'irmas': IRMASDataset, 'openmic': OpenMICDataset, 'both': CombinedDataset}
    train_ds = DatasetClass[args.dataset]('train')
    val_ds   = DatasetClass[args.dataset]('val')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = InstrumentDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = FocalBCE(gamma=2.0)
    scaler    = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    best_f1, no_improve = 0.0, 0
    print(f'Training for up to {args.epochs} epochs...\n')

    for epoch in range(args.epochs):
        t0    = time.time()
        tloss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vm    = evaluate(model, val_loader, criterion, device)
        scheduler.step(vm['loss'])

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train={tloss:.4f} | val={vm['loss']:.4f} | "
            f"f1={vm['f1']:.4f} | prec={vm['precision']:.4f} | "
            f"rec={vm['recall']:.4f} | {time.time()-t0:.0f}s"
        )

        if vm['f1'] > best_f1:
            best_f1, no_improve = vm['f1'], 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1':              best_f1,
            }, CHECKPOINT_PATH)
            print(f'  New best F1={best_f1:.4f} — saved to {CHECKPOINT_PATH}')
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print('Early stopping.')
                break

    print(f'\nTraining complete. Best val F1: {best_f1:.4f}')
    print(f'Checkpoint: {CHECKPOINT_PATH}')


if __name__ == '__main__':
    main()
