"""
src/predict.py — download audio, run inference, print timeline.

Usage:
    python src/predict.py --url "https://www.youtube.com/watch?v=..."
    python src/predict.py --url "https://open.spotify.com/track/..." --threshold 0.4
    python src/predict.py --file /path/to/song.wav
"""

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INSTRUMENT_CLASSES, NUM_CLASSES, CHECKPOINT_PATH,
    OUTPUT_DIR, TEMP_DIR,
    CONFIDENCE_THRESHOLD, MERGE_GAP_SECONDS, MIN_SEGMENT_SECONDS, HOP_SECONDS,
)
from src.model import load_checkpoint
from src.preprocessor import windowed_segments


# ── Downloader (exact from Cell 4) ───────────────────────────────────────────

def download_audio(url: str) -> Path:
    url    = url.strip()
    job_id = uuid.uuid4().hex[:8]
    out    = str(TEMP_DIR / f"{job_id}_%(title)s.%(ext)s")

    if 'spotify' in url:
        r = subprocess.run(
            ['spotdl', '--output', str(TEMP_DIR / f'{job_id}_{{title}}'),
             '--format', 'wav', url],
            capture_output=True, text=True, cwd=str(TEMP_DIR), timeout=120,
        )
    else:
        r = subprocess.run(
            ['yt-dlp', '--extract-audio', '--audio-format', 'wav',
             '--audio-quality', '0', '--output', out,
             '--no-playlist', url],
            capture_output=True, text=True, timeout=120,
        )

    if r.returncode != 0:
        raise RuntimeError(f'Download failed:\n{r.stderr}')

    wavs = sorted(TEMP_DIR.glob(f'{job_id}_*.wav'),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise FileNotFoundError('No .wav file found after download.')
    return wavs[0]


# ── Post-processing (exact from Cell 4) ──────────────────────────────────────

def fmt_time(s):
    return f'{int(s)//60}:{int(s)%60:02d}'


def build_timeline(results, thr=CONFIDENCE_THRESHOLD):
    if not results:
        return []
    active = [
        (t, t + HOP_SECONDS, frozenset(k for k, v in c.items() if v >= thr), dict(c))
        for t, c in results
    ]
    cs, ce, ci, cc = active[0]
    merged = []
    for s, e, instr, confs in active[1:]:
        if instr == ci and s - ce <= MERGE_GAP_SECONDS:
            ce = e
            for k in confs:
                cc[k] = (cc.get(k, 0) + confs[k]) / 2
        else:
            merged.append((cs, ce, ci, cc))
            cs, ce, ci, cc = s, e, instr, dict(confs)
    merged.append((cs, ce, ci, cc))

    out = []
    for s, e, instr, confs in merged:
        if e - s < MIN_SEGMENT_SECONDS or not instr:
            continue
        ac = {k: round(float(confs.get(k, 0)), 3) for k in instr}
        out.append({
            'start':       round(s, 2),
            'end':         round(e, 2),
            'start_fmt':   fmt_time(s),
            'end_fmt':     fmt_time(e),
            'instruments': sorted(instr, key=lambda k: ac.get(k, 0), reverse=True),
            'confidences': ac,
        })
    return out


def print_timeline(segs, title=''):
    print(f'\n  {title}')
    print('-' * 60)
    if not segs:
        print('No instruments detected.')
        return
    for seg in segs:
        t = f"{seg['start_fmt']} - {seg['end_fmt']}"
        i = ', '.join(seg['instruments'])
        c = '  '.join(f"{k}={seg['confidences'][k]:.0%}" for k in seg['instruments'])
        print(f'  {t:<14}  {i}')
        print(f'  {" " * 14}  [{c}]')
        print()
    print('-' * 60)
    all_i = sorted({i for s in segs for i in s['instruments']})
    print(f'Instruments found: {", ".join(all_i)}')


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(audio_path, model, device, threshold=CONFIDENCE_THRESHOLD):
    segs, duration = windowed_segments(audio_path)
    results        = []
    BATCH          = 128
    for i in range(0, len(segs), BATCH):
        batch   = segs[i:i + BATCH]
        times   = [s[0] for s in batch]
        tensors = torch.stack([s[1] for s in batch]).to(device)
        probs   = torch.sigmoid(model(tensors)).cpu().numpy()
        for j, t in enumerate(times):
            results.append((t, {
                INSTRUMENT_CLASSES[k]: float(probs[j, k])
                for k in range(NUM_CLASSES)
            }))
    return build_timeline(results, thr=threshold), duration


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Identify instruments in a song with timestamps')
    parser.add_argument('--url',       type=str,   help='YouTube or Spotify URL')
    parser.add_argument('--file',      type=str,   help='Path to a local audio file')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument('--checkpoint',type=str,   default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error('Provide --url or --file')

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
    print(f'Device: {device}')

    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f'Model loaded  (val F1: {ckpt.get("val_f1", "?")})')

    if args.file:
        audio_path = Path(args.file)
        title      = audio_path.stem
    else:
        print(f'Downloading: {args.url}')
        audio_path = download_audio(args.url)
        title      = re.sub(r'^[a-f0-9]{8}_', '', audio_path.stem)

    print(f'Analysing: {title}')
    t0                 = time.time()
    timeline, duration = run_inference(audio_path, model, device, threshold=args.threshold)
    print(f'Done in {time.time()-t0:.1f}s')

    print_timeline(timeline, title=title)

    safe = re.sub(r'[^\w\s-]', '_', title)[:60].strip()
    out  = OUTPUT_DIR / f'{safe}.json'
    all_i = sorted({i for s in timeline for i in s['instruments']})
    with open(out, 'w') as f:
        json.dump({
            'title':             title,
            'url':               args.url or '',
            'duration':          round(duration, 2),
            'instruments_found': all_i,
            'timeline':          timeline,
        }, f, indent=2)
    print(f'\nResults saved to: {out}')

    if args.file is None and audio_path.exists():
        audio_path.unlink()


if __name__ == '__main__':
    main()
