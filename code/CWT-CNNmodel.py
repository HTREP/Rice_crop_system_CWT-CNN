#!/home/capybara03/pyENV/geo/bin/python3
"""
Sections:
    1.  Imports & device
    2.  Config
    3.  Load raster list / dates / metadata
    4.  Read + rasterize training shapefile
    5.  Extract training time series  (rasterio.sample)
    6.  SG smooth + interp + scipy CWT  ->  (128,128,3) per sample
    7.  Dataset + train/test split
    8.  CNNModel definition
    9.  Training loop  (label smoothing, early stop)
    10. Precompute CuPy CWT kernels
    11. Predict full raster  (chunk-row)
    12. Save output GeoTIFF
"""

import os, re, glob, copy, time, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed when running in tmux
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import cv2
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score,
                              accuracy_score)
import seaborn as sn
from tqdm import tqdm

import cupy as cp
from torch.utils.dlpack import from_dlpack

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING  — writes to both stdout and a log file simultaneously
# =============================================================================
def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = '%(asctime)s | %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(),
        ]
    )

def log(msg: str):
    logging.info(msg)

def section(title: str):
    bar = '=' * 65
    logging.info(bar)
    logging.info(f'  {title}')
    logging.info(bar)

# =============================================================================
# CONFIG
# =============================================================================
S2_FOLDER  = '/home/capybara03/perth/Paper_perth2026/Sentinel-2_final'
S1_FOLDER  = '/home/capybara03/perth/Paper_perth2026/Sentinel-1_final'
SHP_PATH   = '/home/capybara03/perth/Paper_perth2026/SHP/Rice_sample/Rice_sample_N.shp'
TRAIN_ATTR = 'class'

# Each run creates a new RoundN folder so results never overwrite each other
BASE_ROUND_DIR = '/home/capybara03/perth/Paper_perth2026/result/proposed'

def get_next_round_dir(base: str) -> str:
    """Auto-increment: Round1, Round2, ... based on existing subfolders."""
    os.makedirs(base, exist_ok=True)
    existing = [d for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d))
                and re.match(r'^Round\d+$', d)]
    n = max((int(re.search(r'\d+', d).group()) for d in existing), default=0)
    path = os.path.join(base, f'Round{n + 1}')
    os.makedirs(path, exist_ok=True)
    return path

# CWT CONFIG  (identical to create_datasetS1/S2.py)
_DT     = 1
_FS     = 1 / _DT
_FREQ   = np.linspace(1, _FS / 2, 100)
_W      = 6
_WIDTHS = _W * _FS / (2 * _FREQ * np.pi)
IMAGE_SIZE = (128, 128)

# Preprocessing
SG_WINDOW  = 9
SG_ORDER   = 2
MIN_VALID  = 4

# Training
BATCH_SIZE   = 32
MAX_EPOCHS   = 2000
ES_PATIENCE  = 100
LR           = 3e-4
WEIGHT_DECAY = 5e-4
DROPOUT      = 0.4

# Prediction
CHUNK_ROWS = 20
GPU_BATCH  = 2000

CLASS_NAMES  = {0: 'SCR', 1: 'DRC', 2: 'HRC', 3: 'TRC'}
TARGET_NAMES = [CLASS_NAMES[i] for i in range(4)]
RASTER_MAP   = {'1': 1, '2': 2, '3': 3, '4': 4}

# =============================================================================
# HELPERS
# =============================================================================
def parse_date(fname):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(fname))
    return m.group(1) if m else ''


def sg_interp_cwt(days, values, target_days, val_range=None):
    days   = np.asarray(days,   dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if val_range is not None:
        m = (values >= val_range[0]) & (values <= val_range[1])
        days, values = days[m], values[m]
    valid = np.isfinite(values)
    days, values = days[valid], values[valid]
    df = (pd.DataFrame({'d': days, 'v': values})
            .groupby('d').mean().reset_index())
    days, values = df['d'].values, df['v'].values
    if len(days) < MIN_VALID:
        return None
    win = min(SG_WINDOW, len(days))
    if win % 2 == 0: win -= 1
    if win >= SG_ORDER + 2:
        values = sp_signal.savgol_filter(values, window_length=win, polyorder=SG_ORDER)
    fn     = interp1d(days, values, kind='linear',
                      bounds_error=False, fill_value=(values[0], values[-1]))
    interp = fn(target_days).astype(np.float32)
    cwt_result = sp_signal.cwt(interp, sp_signal.morlet2, _WIDTHS, w=_W)
    H, W = IMAGE_SIZE
    return cv2.resize(np.abs(cwt_result), (W, H),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


class CWTDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        arr = self.arrays[idx].copy().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr /= (np.max(arr) + 1e-8)
        arr = arr.transpose(2, 0, 1)
        return (torch.tensor(arr, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


class CNNModel(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.fc(self.conv(x))


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(device)
            preds = model(X_b).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)


def _precompute_cwt_kernels(T, widths, w):
    kernels_real, kernels_imag, starts = [], [], []
    for width in widths:
        M   = min(int(10 * width) + 1, T)
        x   = (np.arange(0, M, dtype=np.float64) - (M - 1.0) / 2) / width
        wav = np.sqrt(1.0 / width) * (
            np.exp(1j * w * x) * np.exp(-0.5 * x**2) * (np.pi ** -0.25)
        )
        kern = np.conj(wav)[::-1]
        kernels_real.append(kern.real.astype(np.float32))
        kernels_imag.append(kern.imag.astype(np.float32))
        starts.append((M - 1) // 2)
    return kernels_real, kernels_imag, starts


def cwt_batch_cupy(signals_np, kr_ffts, ki_ffts, starts, n_fft, T, image_size):
    H, W      = image_size
    n_scales  = len(kr_ffts)
    sig_gpu   = cp.asarray(signals_np, dtype=cp.float32)
    N         = sig_gpu.shape[0]
    sig_fft   = cp.fft.rfft(sig_gpu, n=n_fft, axis=1)
    scalogram = cp.empty((N, n_scales, T), dtype=cp.float32)
    for i in range(n_scales):
        r  = cp.fft.irfft(sig_fft * kr_ffts[i], n=n_fft, axis=1)
        im = cp.fft.irfft(sig_fft * ki_ffts[i], n=n_fft, axis=1)
        scalogram[:, i, :] = cp.sqrt(r[:, starts[i]:starts[i] + T] ** 2 +
                                      im[:, starts[i]:starts[i] + T] ** 2)
    scal_t  = from_dlpack(scalogram.toDlpack()).unsqueeze(1).float()
    resized = F.interpolate(scal_t, size=(H, W), mode='area')
    return resized.squeeze(1)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    # ── Round folder — all outputs go here, nothing overwrites previous runs ──
    ROUND_DIR            = get_next_round_dir(BASE_ROUND_DIR)
    classification_image = os.path.join(ROUND_DIR, 'predicted_raster_CNN_CWT.tif')
    model_output         = os.path.join(ROUND_DIR, 'pretrained_model_CNN_CWT.pth')
    train_log_csv        = os.path.join(ROUND_DIR, 'training_log_CNN_CWT.csv')
    run_log              = os.path.join(ROUND_DIR, 'run_CNN_CWT.log')
    plot_dir             = ROUND_DIR   # plots saved directly in round folder

    setup_logging(run_log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_start = time.time()
    section('CWT-CNNmodel  |  start')
    log(f'Round dir : {ROUND_DIR}')
    log(f'Device    : {device}')

    # ── 3. Raster list & dates ────────────────────────────────────────────────
    section('3. Load raster list & metadata')
    s2_files = sorted(glob.glob(os.path.join(S2_FOLDER, '*.tif')), key=parse_date)
    s1_files = sorted(glob.glob(os.path.join(S1_FOLDER, '*.tif')), key=parse_date)

    S2_START = datetime.strptime(parse_date(s2_files[0]), '%Y-%m-%d')
    S1_START = datetime.strptime(parse_date(s1_files[0]), '%Y-%m-%d')

    s2_days = np.array([(datetime.strptime(parse_date(f), '%Y-%m-%d') - S2_START).days
                        for f in s2_files], dtype=np.float64)
    s1_days = np.array([(datetime.strptime(parse_date(f), '%Y-%m-%d') - S1_START).days
                        for f in s1_files], dtype=np.float64)

    T_S2      = int(s2_days[-1]) + 1
    T_S1      = int(s1_days[-1]) + 1
    TARGET_S2 = np.arange(T_S2, dtype=np.float64)
    TARGET_S1 = np.arange(T_S1, dtype=np.float64)

    with rasterio.open(s2_files[0]) as src:
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        height    = src.height
        width     = src.width

    log(f'S2: {len(s2_files)} files | T_S2={T_S2}')
    log(f'S1: {len(s1_files)} files | T_S1={T_S1}')
    log(f'Raster: {width} x {height}')

    # ── 4. Shapefile ──────────────────────────────────────────────────────────
    section('4. Read & rasterize shapefile')
    gdf   = gpd.read_file(SHP_PATH)
    gdf_r = gdf.copy()
    gdf_r[TRAIN_ATTR] = gdf_r[TRAIN_ATTR].map(RASTER_MAP)
    if gdf_r.crs != crs:
        gdf_r = gdf_r.to_crs(crs)

    roi = rasterize(
        [(geom, val) for geom, val in zip(gdf_r.geometry, gdf_r[TRAIN_ATTR])],
        out_shape=(height, width), transform=transform, fill=0, dtype='uint8'
    )
    log(f'Shapefile: {len(gdf)} points | classes: {sorted(gdf[TRAIN_ATTR].unique())}')
    log(f'Label counts: { {CLASS_NAMES[v-1]: int((roi==v).sum()) for v in [1,2,3,4]} }')

    # ── 5. Extract training time series ──────────────────────────────────────
    section('5. Extract training time series')
    coords_xy = [(geom.x, geom.y) for geom in gdf_r.geometry]
    n_points  = len(coords_xy)

    evi_ts  = np.full((n_points, len(s2_files)), np.nan, dtype=np.float32)
    ndre_ts = np.full((n_points, len(s2_files)), np.nan, dtype=np.float32)
    vh_ts   = np.full((n_points, len(s1_files)), np.nan, dtype=np.float32)

    log('Reading S2 (EVI + NDRE) ...')
    for i, f in enumerate(tqdm(s2_files, desc='S2')):
        with rasterio.open(f) as src:
            vals = np.array(list(src.sample(coords_xy)), dtype=np.float32)
            evi_ts[:, i]  = vals[:, 0]
            ndre_ts[:, i] = vals[:, 1]

    log('Reading S1 (VH) ...')
    for i, f in enumerate(tqdm(s1_files, desc='S1')):
        with rasterio.open(f) as src:
            vals = np.array(list(src.sample(coords_xy)), dtype=np.float32)
            vh_ts[:, i] = vals[:, 0]

    labels_raw = gdf_r[TRAIN_ATTR].values.astype(int)
    log(f'Time series — EVI: {evi_ts.shape}  VH: {vh_ts.shape}')

    # ── 6. Preprocess + CWT ──────────────────────────────────────────────────
    section('6. SG smooth + interp + CWT  (training samples)')
    stacked_list, labels_list, skipped = [], [], []

    for idx in tqdm(range(n_points), desc='CWT'):
        evi_scal  = sg_interp_cwt(s2_days, evi_ts[idx],  TARGET_S2, val_range=(0.0, 1.0))
        ndre_scal = sg_interp_cwt(s2_days, ndre_ts[idx], TARGET_S2)
        vh_scal   = sg_interp_cwt(s1_days, vh_ts[idx],   TARGET_S1)
        if any(v is None for v in (evi_scal, ndre_scal, vh_scal)):
            skipped.append(idx)
            continue
        stacked_list.append(np.stack([evi_scal, ndre_scal, vh_scal], axis=-1))
        labels_list.append(labels_raw[idx] - 1)

    if skipped:
        log(f'Skipped {len(skipped)} points (too few valid obs): {skipped}')

    labels_arr = np.array(labels_list, dtype=np.int64)
    log(f'Samples: {len(stacked_list)} | scalogram: {stacked_list[0].shape}')
    log(f'Class distribution: { {CLASS_NAMES[v]: int((labels_arr==v).sum()) for v in range(4)} }')

    # ── 7. Dataset + split ────────────────────────────────────────────────────
    section('7. Dataset + train/test split')
    indices = np.arange(len(stacked_list))
    tr_idx, te_idx, y_train_all, y_test = train_test_split(
        indices, labels_arr, test_size=0.2, stratify=labels_arr, random_state=42
    )

    train_ds = CWTDataset([stacked_list[i] for i in tr_idx], y_train_all)
    test_ds  = CWTDataset([stacked_list[i] for i in te_idx], y_test)

    class_counts   = np.bincount(y_train_all, minlength=4).astype(float)
    sample_weights = np.array([1.0 / class_counts[y] for y in y_train_all], dtype=np.float32)
    sampler        = WeightedRandomSampler(sample_weights,
                                           num_samples=len(sample_weights),
                                           replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    log(f'Train: {len(train_ds)}  Test: {len(test_ds)}')
    log(f'Class counts (train): { {CLASS_NAMES[i]: int(c) for i,c in enumerate(class_counts)} }')

    # ── 8-9. Model + training ─────────────────────────────────────────────────
    section('8-9. CNNModel + Training')
    model = CNNModel(num_classes=4, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'CNNModel ready | trainable params: {n_params:,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6
    )

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    best_state = None
    train_losses, precisions, recalls, f1_scores = [], [], [], []

    log(f'Training (max {MAX_EPOCHS} epochs, early-stop patience={ES_PATIENCE}) ...')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss    = running_loss / max(1, len(train_loader))
        all_targets = [int(y) for _, yb in test_loader for y in yb.numpy()]
        all_preds   = evaluate(model, test_loader, device)

        p  = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        r  = recall_score   (all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score       (all_targets, all_preds, average='macro', zero_division=0)
        train_losses.append(avg_loss); precisions.append(p); recalls.append(r); f1_scores.append(f1)

        scheduler.step(f1)
        lr_now = optimizer.param_groups[0]['lr']

        if f1 > best_f1:
            best_f1, best_epoch, no_improve = f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1

        log(f'Epoch {epoch:04d}/{MAX_EPOCHS} | Loss: {avg_loss:.4f} | '
            f'P: {p:.3f} R: {r:.3f} F1: {f1:.3f} | '
            f'LR: {lr_now:.2e} | best F1: {best_f1:.3f} @ ep{best_epoch}')

        if no_improve >= ES_PATIENCE:
            log(f'Early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f'Restored best model: epoch {best_epoch}  F1 = {best_f1:.4f}')

    # ── 10. Training curves ───────────────────────────────────────────────────
    section('10. Training curves')
    epochs_range = np.arange(1, len(train_losses) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    for ax, data, title, color in zip(axs,
        [train_losses, precisions, recalls, f1_scores],
        ['Training Loss', 'Precision', 'Recall', 'F1-Score'],
        ['blue', 'green', 'orange', 'red']):
        ax.plot(epochs_range, data, color=color, lw=2)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(True, linestyle='--', alpha=0.5)
    axs[3].axhline(best_f1, color='red', linestyle='--', alpha=0.5, label=f'best={best_f1:.3f}')
    axs[3].legend()
    plt.tight_layout()
    curve_path = os.path.join(plot_dir, 'training_curves_CNN_CWT.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    log(f'Training curves saved -> {curve_path}')

    pd.DataFrame({'epoch': epochs_range, 'loss': train_losses,
                  'precision': precisions, 'recall': recalls, 'f1': f1_scores
                 }).to_csv(train_log_csv, index=False, sep='\t')
    log(f'Training log saved -> {train_log_csv}')

    # ── 11. Final evaluation ──────────────────────────────────────────────────
    section('11. Final Evaluation')
    all_targets = [int(y) for _, yb in test_loader for y in yb.numpy()]
    all_preds   = evaluate(model, test_loader, device)

    report_str = classification_report(all_targets, all_preds,
                                        target_names=TARGET_NAMES, digits=4)
    log('\n' + report_str)
    log(f'Test accuracy : {accuracy_score(all_targets, all_preds):.4f}')
    log(f'Macro F1      : {f1_score(all_targets, all_preds, average="macro"):.4f}')

    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
    ax.set_title('CNN-CWT Confusion Matrix')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, 'confusion_matrix_CNN_CWT.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log(f'Confusion matrix saved -> {cm_path}')

    report_dict = classification_report(all_targets, all_preds,
                                         target_names=TARGET_NAMES, digits=4, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(ROUND_DIR, 'classification_report_CNN_CWT.csv'))
    pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_csv(
        os.path.join(ROUND_DIR, 'confusion_matrix_CNN_CWT.csv'))

    torch.save(model.state_dict(), model_output)
    log(f'Model saved -> {model_output}')
    log(f'Reports saved -> {ROUND_DIR}')

    # ── 12. CuPy CWT kernels ─────────────────────────────────────────────────
    section('12. Precompute CuPy CWT kernels')
    kr_s2_np, ki_s2_np, starts_s2 = _precompute_cwt_kernels(T_S2, _WIDTHS, _W)
    kr_s1_np, ki_s1_np, starts_s1 = _precompute_cwt_kernels(T_S1, _WIDTHS, _W)

    N_FFT_S2 = int(2 ** np.ceil(np.log2(T_S2 + max(len(k) for k in kr_s2_np))))
    N_FFT_S1 = int(2 ** np.ceil(np.log2(T_S1 + max(len(k) for k in kr_s1_np))))

    kr_s2_gpu = [cp.fft.rfft(cp.asarray(k), n=N_FFT_S2) for k in kr_s2_np]
    ki_s2_gpu = [cp.fft.rfft(cp.asarray(k), n=N_FFT_S2) for k in ki_s2_np]
    kr_s1_gpu = [cp.fft.rfft(cp.asarray(k), n=N_FFT_S1) for k in kr_s1_np]
    ki_s1_gpu = [cp.fft.rfft(cp.asarray(k), n=N_FFT_S1) for k in ki_s1_np]

    log(f'CuPy CWT ready | N_FFT_S2={N_FFT_S2} | N_FFT_S1={N_FFT_S1} | scales=100')

    # ── 13. Predict full raster ───────────────────────────────────────────────
    section('13. Predict full raster  (chunk-row + CuPy CWT)')
    model.eval()
    class_prediction = np.zeros((height, width), dtype=np.uint16)

    s2_srcs = [rasterio.open(f) for f in s2_files]
    s1_srcs = [rasterio.open(f) for f in s1_files]
    n_chunks = (height + CHUNK_ROWS - 1) // CHUNK_ROWS

    try:
        for chunk_i, row_start in enumerate(range(0, height, CHUNK_ROWS), 1):
            row_end = min(row_start + CHUNK_ROWS, height)
            h_chunk = row_end - row_start
            n_pix   = h_chunk * width
            window  = rasterio.windows.Window(0, row_start, width, h_chunk)

            evi_px  = np.stack([src.read(1, window=window) for src in s2_srcs], axis=0
                               ).reshape(len(s2_files), -1).T.astype(np.float32)
            ndre_px = np.stack([src.read(2, window=window) for src in s2_srcs], axis=0
                               ).reshape(len(s2_files), -1).T.astype(np.float32)
            vh_px   = np.stack([src.read(1, window=window) for src in s1_srcs], axis=0
                               ).reshape(len(s1_files), -1).T.astype(np.float32)

            valid_mask  = np.isfinite(evi_px).sum(axis=1) >= MIN_VALID
            valid_mask &= np.isfinite(vh_px).sum(axis=1)  >= MIN_VALID

            if chunk_i % 50 == 0 or chunk_i == n_chunks:
                log(f'  Chunk {chunk_i}/{n_chunks} | row {row_start}-{row_end} '
                    f'| valid px: {valid_mask.sum():,}/{n_pix:,}')

            if not valid_mask.any():
                continue

            evi_px  = np.nan_to_num(evi_px,  nan=0.0, posinf=0.0, neginf=0.0)
            ndre_px = np.nan_to_num(ndre_px, nan=0.0, posinf=0.0, neginf=0.0)
            vh_px   = np.nan_to_num(vh_px,   nan=0.0, posinf=0.0, neginf=0.0)

            evi_v  = evi_px [valid_mask]
            ndre_v = ndre_px[valid_mask]
            vh_v   = vh_px  [valid_mask]
            N_v    = evi_v.shape[0]

            evi_sm  = sp_signal.savgol_filter(evi_v,  SG_WINDOW, SG_ORDER, axis=1)
            ndre_sm = sp_signal.savgol_filter(ndre_v, SG_WINDOW, SG_ORDER, axis=1)
            vh_sm   = sp_signal.savgol_filter(vh_v,   SG_WINDOW, SG_ORDER, axis=1)

            evi_int  = interp1d(s2_days, evi_sm,  kind='linear', axis=1,
                                 bounds_error=False, fill_value='extrapolate')(TARGET_S2).astype(np.float32)
            ndre_int = interp1d(s2_days, ndre_sm, kind='linear', axis=1,
                                 bounds_error=False, fill_value='extrapolate')(TARGET_S2).astype(np.float32)
            vh_int   = interp1d(s1_days, vh_sm,   kind='linear', axis=1,
                                 bounds_error=False, fill_value='extrapolate')(TARGET_S1).astype(np.float32)

            pred_valid = np.zeros(N_v, dtype=np.uint16)
            for b_s in range(0, N_v, GPU_BATCH):
                b_e = min(b_s + GPU_BATCH, N_v)
                sl  = slice(b_s, b_e)
                B   = b_e - b_s

                evi_cwt  = cwt_batch_cupy(evi_int[sl],  kr_s2_gpu, ki_s2_gpu,
                                          starts_s2, N_FFT_S2, T_S2, IMAGE_SIZE)
                ndre_cwt = cwt_batch_cupy(ndre_int[sl], kr_s2_gpu, ki_s2_gpu,
                                          starts_s2, N_FFT_S2, T_S2, IMAGE_SIZE)
                vh_cwt   = cwt_batch_cupy(vh_int[sl],   kr_s1_gpu, ki_s1_gpu,
                                          starts_s1, N_FFT_S1, T_S1, IMAGE_SIZE)

                imgs  = torch.stack([evi_cwt, ndre_cwt, vh_cwt], dim=1).float()
                imgs  = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
                max_v = imgs.view(B, -1).max(dim=1).values.view(B, 1, 1, 1) + 1e-8
                imgs /= max_v

                with torch.no_grad():
                    preds = model(imgs).argmax(dim=1).cpu().numpy().astype(np.uint16)
                pred_valid[b_s:b_e] = preds + 1

            pred_flat = np.zeros(n_pix, dtype=np.uint16)
            pred_flat[valid_mask] = pred_valid
            class_prediction[row_start:row_end, :] = pred_flat.reshape(h_chunk, width)

    finally:
        for src in s2_srcs + s1_srcs:
            src.close()

    log('Full prediction DONE.')

    # ── 14. Save GeoTIFF ─────────────────────────────────────────────────────
    section('14. Save output GeoTIFF')
    profile.update({'count': 1, 'dtype': 'uint16', 'nodata': 0})
    with rasterio.open(classification_image, 'w', **profile) as dst:
        dst.write(class_prediction, 1)
    log(f'Classification saved -> {classification_image}')

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    section(f'DONE  |  total time: {elapsed/3600:.2f} h  ({elapsed:.0f} s)')
    log(f'All outputs -> {ROUND_DIR}')
