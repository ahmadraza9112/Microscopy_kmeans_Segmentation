"""Reproducible introductory microscopy segmentation benchmark.

Real microscopy: scikit-image data.cell() (no reference mask).
Synthetic microscopy-style phantoms: reproducibly generated masks.
K-means is UNSUPERVISED; no neural networks, no fitting on ground truth.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')  # Also works on headless machines and GitHub runners.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
from scipy.ndimage import gaussian_filter
from skimage import data, filters, morphology, transform
from skimage.draw import ellipse
from sklearn.cluster import KMeans

SEED = 42
METHODS = ('Otsu', 'KMeans-2', 'KMeans-3')


def synthetic_scene(seed: int, condition: str = 'clean') -> tuple[np.ndarray, np.ndarray]:
    """Make microscopy-*style* bright ellipses with exact binary reference mask.

    These are synthetic shapes, NOT biological specimens or manual annotations.
    """
    if condition not in ('clean', 'noisy'):
        raise ValueError('condition must be clean or noisy')
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    truth = np.zeros((h, w), dtype=bool)
    signal = np.zeros((h, w), dtype=float)
    centers = [(62, 64), (65, 190), (178, 67), (186, 187)]
    for cy, cx in centers:
        cy += int(rng.integers(-7, 8))
        cx += int(rng.integers(-7, 8))
        ry = int(rng.integers(17, 27))
        rx = int(rng.integers(16, 27))
        rr, cc = ellipse(cy, cx, ry, rx, shape=(h, w))
        truth[rr, cc] = True
        signal[rr, cc] = rng.uniform(0.63, 0.89)

    yy, xx = np.indices((h, w))
    spatial_gradient = (xx / (w-1) - 0.5) * (0.08 if condition == 'clean' else 0.22)
    background = 0.15 + spatial_gradient + 0.02 * np.sin(yy / 35.0)
    blurred_cells = gaussian_filter(signal, sigma=1.15)
    noise_sigma = 0.025 if condition == 'clean' else 0.09
    image = background + blurred_cells + rng.normal(0, noise_sigma, (h, w))
    return np.clip(image, 0.0, 1.0).astype(np.float32), truth


def real_cell_image() -> np.ndarray:
    """Bundled microscopy image resized for a fast CPU-based demonstration."""
    image = data.cell()
    resized = transform.resize(image, (330, 275), anti_aliasing=True)
    return np.asarray(resized, dtype=np.float32)


def postprocess(mask: np.ndarray) -> np.ndarray:
    """Identical morphological cleanup for all methods."""
    result = morphology.area_opening(mask.astype(bool), area_threshold=35)
    result = morphology.closing(result, footprint=morphology.disk(2))
    result = morphology.area_closing(result, area_threshold=40)
    return np.asarray(result, dtype=bool)


def segment(image: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    """Generate a binary foreground mask (bright cells), tracking wall-clock runtime.

    KMeans learns brightness centers from every fourth smoothed pixel.
    Brightest cluster is the foreground for BOTH real and synthetic images.
    There is no use of reference masks in model fitting or selection.
    """
    t0 = perf_counter()
    smooth = filters.gaussian(image, sigma=1.0)
    if method == 'Otsu':
        raw = smooth > filters.threshold_otsu(smooth)
    elif method in ('KMeans-2', 'KMeans-3'):
        clusters = 2 if method == 'KMeans-2' else 3
        pixels = smooth.reshape(-1, 1).astype(np.float32)
        sample = pixels[::4]  # Deterministic spatial subsampling; no random data split.
        model = KMeans(n_clusters=clusters, random_state=SEED, n_init=5, max_iter=100)
        model.fit(sample)
        foreground_class = int(np.argmax(model.cluster_centers_.ravel()))
        raw = (model.predict(pixels).reshape(image.shape) == foreground_class)
    else:
        raise ValueError(f'Unknown method: {method}')
    result = postprocess(raw)
    return result, perf_counter() - t0


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Binary pixel metrics. Only apply when a genuine or simulated mask exists."""
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}


def plot_real(image: np.ndarray, masks: dict[str, np.ndarray], destination: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout='constrained')
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Real microscopy image')
    for ax, method in zip(axes[1:], METHODS):
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.contour(masks[method], levels=[0.5], colors=['#ff6d00'], linewidths=1.5)
        ax.set_title(method + ' (outline)')
    for ax in axes: ax.axis('off')
    fig.suptitle('Real specimen: qualitative comparison only (no reference mask)', fontsize=11)
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_synthetic(image: np.ndarray, truth: np.ndarray, masks: dict[str, np.ndarray], destination: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), layout='constrained')
    views = [('Synthetic microscopy-like image', image, 'gray'), ('Reference mask', truth, 'gray')]
    views.extend((name, masks[name], 'gray') for name in METHODS)
    for ax, (title, img, cmap) in zip(axes, views):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle('Controlled synthetic illustration (ground truth available)', fontsize=11)
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(table: pd.DataFrame, destination: Path) -> None:
    means = table.groupby(['condition', 'method'])['dice'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout='constrained')
    groups = ('clean', 'noisy')
    x = np.arange(len(groups))
    for j, method in enumerate(METHODS):
        rows = means[means.method == method].set_index('condition').loc[list(groups)]
        ax.bar(x + (j - 1) * 0.24, rows['mean'], width=0.22,
               yerr=rows['std'].fillna(0), capsize=3, label=method)
    ax.set_xticks(x, ['Clean synthetic', 'Noisy synthetic'])
    ax.set_ylabel('Mean Dice (± sample SD)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Pixel-level agreement with generated reference masks (n=3 per condition)')
    ax.legend()
    fig.savefig(destination, dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_all(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root).resolve()
    (root / 'figures').mkdir(exist_ok=True, parents=True)
    (root / 'results').mkdir(exist_ok=True, parents=True)

    real = real_cell_image()
    real_masks = {}
    real_rows = []
    for method in METHODS:
        mask, seconds = segment(real, method)
        real_masks[method] = mask
        real_rows.append({'method': method, 'foreground_fraction': float(mask.mean()),
                          'runtime_seconds': round(seconds, 4)})
    real_table = pd.DataFrame(real_rows)
    real_table.to_csv(root / 'results' / 'real_microscopy_descriptive.csv', index=False)
    plot_real(real, real_masks, root / 'figures' / 'real_microscopy_comparison.png')

    rows = []
    for condition in ('clean', 'noisy'):
        for example in range(3):
            image, truth = synthetic_scene(seed=100 + example, condition=condition)
            predictions = {}
            for method in METHODS:
                pred, secs = segment(image, method)
                predictions[method] = pred
                rows.append({'condition': condition, 'sample': example + 1, 'method': method,
                             **binary_metrics(pred, truth),
                             'foreground_fraction': float(pred.mean()),
                             'runtime_seconds': round(secs, 4)})
            if example == 0:
                plot_synthetic(image, truth, predictions,
                               root / 'figures' / f'synthetic_{condition}_comparison.png')
    table = pd.DataFrame(rows)
    table.to_csv(root / 'results' / 'synthetic_pixel_metrics.csv', index=False,
                 float_format='%.6f')
    plot_metrics(table, root / 'figures' / 'synthetic_dice_comparison.png')

    summary = (table.groupby(['condition', 'method'], sort=False)
               .agg(mean_dice=('dice', 'mean'), sd_dice=('dice', 'std'),
                    mean_iou=('iou', 'mean'), mean_precision=('precision', 'mean'),
                    mean_recall=('recall', 'mean'),
                    mean_runtime_s=('runtime_seconds', 'mean'))
               .reset_index())
    summary.to_csv(root / 'results' / 'summary.csv', index=False, float_format='%.6f')
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'scikit-image': skimage.__version__, 'scikit-learn': sklearn.__version__,
                   'scipy': scipy.__version__, 'pandas': pd.__version__,
                   'matplotlib': matplotlib.__version__, 'random_seed': SEED,
                   'synthetic_samples_per_condition': 3,
                   'real_ground_truth': False}
    (root / 'results' / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print('\nSynthetic validation (mean across 3 examples per condition):')
    print(summary[['condition', 'method', 'mean_dice', 'mean_iou']].round(3).to_string(index=False))
    print('\nReal sample (DESCRIPTIVE; no ground truth):')
    print(real_table.round(3).to_string(index=False))
    return summary, real_table
