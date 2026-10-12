"""Model independent alignment/uniformity metrics and seven-panel figure."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def validate_features(features, labels, sample_ids):
    z = np.asarray(features)
    y = np.asarray(labels)
    ids = np.asarray(sample_ids)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError('features must have shape [N,D] with N,D >= 1')
    if y.ndim != 1 or ids.ndim != 1 or len(y) != len(z) or len(ids) != len(z):
        raise ValueError('features, labels and sample_ids must have matching rows')
    if z.dtype.kind not in 'iuf' or not np.isfinite(z).all():
        raise ValueError('features must contain finite real numbers')
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1]).all():
        raise ValueError('labels must be integer 0=benign, 1=malicious')
    if ids.dtype.kind not in 'US' or any(not str(s).strip() or str(s) != str(s).strip() for s in ids):
        raise ValueError('sample_ids must be nonempty strings without surrounding whitespace')
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('duplicate sample_id')
    return z, y.astype(np.int64), ids.astype(str)


def normalized_features(features):
    z = np.array(features, dtype=np.float64, copy=True)
    norms = np.linalg.norm(z, axis=1)
    z /= np.maximum(norms[:, None], 1e-12)
    return z, int(np.count_nonzero(norms <= 1e-12))


def compute_metrics(features, labels, align_alpha=2.0, uniform_t=2.0, pair_block_size=256):
    if align_alpha <= 0 or uniform_t <= 0 or pair_block_size < 1:
        raise ValueError('align_alpha, uniform_t and pair_block_size must be positive')
    z = np.asarray(features)
    y = np.asarray(labels)
    if z.ndim != 2 or z.shape[0] != len(y) or z.shape[1] == 0 or not np.isfinite(z).all():
        raise ValueError('invalid feature matrix or labels')
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1]).all():
        raise ValueError('labels must be integer 0 or 1')
    u, zero_count = normalized_features(z)
    bins = np.linspace(0, 2, 31)
    same_hist = np.zeros(len(bins) - 1, dtype=np.int64)
    different_hist = np.zeros(len(bins) - 1, dtype=np.int64)
    class_power_sums = np.zeros(2, dtype=np.float64)
    class_pair_counts = np.zeros(2, dtype=np.int64)
    same_distance_sum = 0.0
    same_pair_count = 0
    different_pair_count = 0
    pair_log_sum = -np.inf
    n_total = len(y)
    # Visit every unordered pair exactly once. Only one pair block is resident.
    for first in range(0, n_total, pair_block_size):
        first_end = min(first + pair_block_size, n_total)
        first_labels = y[first:first_end]
        for second in range(first, n_total, pair_block_size):
            second_end = min(second + pair_block_size, n_total)
            second_labels = y[second:second_end]
            sq_matrix = cdist(u[first:first_end], u[second:second_end], metric='sqeuclidean')
            if first == second:
                row, col = np.triu_indices(len(first_labels), k=1)
                squared = sq_matrix[row, col]
                left_labels, right_labels = first_labels[row], second_labels[col]
            else:
                squared = sq_matrix.ravel()
                left_labels = np.repeat(first_labels, len(second_labels))
                right_labels = np.tile(second_labels, len(first_labels))
            if not len(squared):
                continue
            distances = np.sqrt(np.maximum(squared, 0.0))
            pair_log_sum = np.logaddexp(pair_log_sum, logsumexp(-uniform_t * squared))
            same = left_labels == right_labels
            same_distances = distances[same]
            different_distances = distances[~same]
            same_pair_count += len(same_distances)
            different_pair_count += len(different_distances)
            same_distance_sum += float(np.sum(same_distances))
            # Clipping only counters floating error at the mathematical L2 limit 2.
            same_hist += np.histogram(np.minimum(same_distances, 2.0), bins=bins)[0]
            different_hist += np.histogram(np.minimum(different_distances, 2.0), bins=bins)[0]
            for label in (0, 1):
                class_distances = distances[same & (left_labels == label)]
                class_pair_counts[label] += len(class_distances)
                class_power_sums[label] += float(np.sum(class_distances ** align_alpha))
    class_alignments = []
    valid_anchors = 0
    weighted_alignment = 0.0
    for label in (0, 1):
        n = int(np.count_nonzero(y == label))
        value = float(class_power_sums[label] / class_pair_counts[label]) if n >= 2 else None
        class_alignments.append(value)
        if value is not None:
            weighted_alignment += n * value
            valid_anchors += n
    alignment = weighted_alignment / valid_anchors if valid_anchors else None
    pair_count = same_pair_count + different_pair_count
    uniformity = float(pair_log_sum - math.log(pair_count)) if pair_count else None
    return {
        'alignment': alignment,
        'uniformity': uniformity,
        'exp_uniformity': math.exp(uniformity) if uniformity is not None else None,
        'benign_alignment': class_alignments[0],
        'malicious_alignment': class_alignments[1],
        'same_class_pair_mean_l2': same_distance_sum / same_pair_count if same_pair_count else None,
        'sample_count': int(len(y)),
        'benign_count': int(np.count_nonzero(y == 0)),
        'malicious_count': int(np.count_nonzero(y == 1)),
        'pair_count': pair_count,
        'same_class_pair_count': same_pair_count,
        'different_class_pair_count': different_pair_count,
        'valid_alignment_anchor_count': valid_anchors,
        'zero_vector_count': zero_count,
    }, u, same_hist, different_hist


def project_circle(u, labels):
    if min(u.shape) < 2:
        raise ValueError('seven-panel projection requires N,D >= 2')
    # Eigenvectors of the smaller uncentered Gram/covariance matrix give the
    # exact SVD display subspace without storing a full left singular matrix.
    if u.shape[1] > u.shape[0]:
        gram = u @ u.T
        eigenvalues, left = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][:2]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        basis = np.zeros((u.shape[1], 2), dtype=np.float64)
        for j in range(2):
            if singular[j] > 1e-12:
                basis[:, j] = u.T @ left[:, order[j]] / singular[j]
        for j in range(2):
            if np.linalg.norm(basis[:, j]) <= 1e-12:
                # Complete a deterministic orthonormal basis without a D x D array.
                pivot = int(np.argmin(np.sum(basis[:, :j] ** 2, axis=1)))
                candidate = np.zeros(u.shape[1])
                candidate[pivot] = 1.0
                candidate -= basis[:, :j] @ (basis[:, :j].T @ candidate)
                basis[:, j] = candidate / np.linalg.norm(candidate)
    else:
        covariance = u.T @ u
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]].copy()
    for j in range(2):
        pivot = np.argmax(np.abs(basis[:, j]))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    projected = u @ basis
    norms = np.linalg.norm(projected, axis=1)
    valid = norms > 1e-12
    circle = projected / np.maximum(norms[:, None], 1e-12)
    return basis, circle, valid, {
        'projection_degenerate_count': int(np.count_nonzero(~valid)),
        'benign_projection_degenerate_count': int(np.count_nonzero(~valid & (labels == 0))),
        'malicious_projection_degenerate_count': int(np.count_nonzero(~valid & (labels == 1))),
        'projected_valid_count': int(np.count_nonzero(valid)),
    }


def _circle_density(points, grid, bandwidth=0.10, chunk_size=256):
    result = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        squared = np.sum((grid[:, None, :] - chunk[None, :, :]) ** 2, axis=-1)
        result += np.exp(-squared / (2 * bandwidth ** 2)).sum(axis=1)
    return result / (len(points) * 2 * math.pi * bandwidth ** 2)


def _angle_density(angles, grid, kappa=8.0, chunk_size=256):
    result = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(angles), chunk_size):
        result += np.exp(kappa * np.cos(grid[:, None] - angles[None, start:start + chunk_size])).sum(axis=1)
    return result / (len(angles) * 2 * math.pi * np.i0(kappa))


def draw_figure(output_dir, metrics, same_hist, different_hist, circle, valid, labels, model_name, representation, dimension):
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 4, height_ratios=[3, 1], width_ratios=[1.3, 1, 1, 1])
    ax_dist = fig.add_subplot(grid_spec[:, 0])
    bins = np.linspace(0, 2, 31)
    width = bins[1] - bins[0]
    if metrics['same_class_pair_count']:
        ax_dist.bar(bins[:-1], same_hist / (same_hist.sum() * width), width=width, align='edge',
                    alpha=0.45, color='#2678ad', label='Same class')
        ax_dist.axvline(metrics['same_class_pair_mean_l2'], color='#2678ad', ls='--', label='Same-class pair mean')
    if metrics['different_class_pair_count']:
        ax_dist.bar(bins[:-1], different_hist / (different_hist.sum() * width), width=width, align='edge',
                    alpha=0.45, color='#dd8452', label='Different class')
    ax_dist.set(xlim=(0, 2), xlabel='Original-space L2 distance', ylabel='Pair density', title='Alignment: pair distances')
    ax_dist.legend(fontsize=8)

    xy = np.linspace(-1.2, 1.2, 80)
    xx, yy = np.meshgrid(xy, xy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    angle_grid = np.linspace(-math.pi, math.pi, 256)
    phi = np.linspace(-math.pi, math.pi, 256)
    for column, (title, selection) in enumerate((
        ('All samples', np.ones(len(labels), dtype=bool)),
        ('Benign (0)', labels == 0),
        ('Malicious (1)', labels == 1),
    ), start=1):
        points = circle[selection & valid]
        top = fig.add_subplot(grid_spec[0, column])
        top.set_title(f'{title} (n={int(np.count_nonzero(selection))})', fontsize=10)
        if len(points):
            density = _circle_density(points, grid).reshape(xx.shape)
            top.contourf(xx, yy, density, levels=12, cmap='YlGnBu')
        else:
            top.text(0.5, 0.5, 'No valid points', transform=top.transAxes, ha='center', va='center')
        top.plot(np.cos(phi), np.sin(phi), color='gray', lw=0.8)
        top.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        top.set_aspect('equal')

        bottom = fig.add_subplot(grid_spec[1, column])
        if len(points):
            angles = np.arctan2(points[:, 1], points[:, 0])
            density = _angle_density(angles, angle_grid)
            bottom.plot(angle_grid, density, color='#246a9b', lw=1.2)
            bottom.fill_between(angle_grid, density, color='#246a9b', alpha=0.20)
        else:
            bottom.text(0.5, 0.5, 'No valid points', transform=bottom.transAxes, ha='center', va='center')
        bottom.axhline(1 / (2 * math.pi), color='gray', ls='--', lw=0.8)
        bottom.set(xlim=(-math.pi, math.pi), ylim=(0, None), xticks=[-math.pi, 0, math.pi],
                   xticklabels=['-pi', '0', 'pi'], xlabel='Angle (radians)', ylabel='Density')

    def fmt(value):
        return 'undefined' if value is None else f'{value:.4f}'

    fig.suptitle(f'{model_name} | Representation={representation} | D={dimension}\n'
                 f'high-D alignment={fmt(metrics["alignment"])}, uniformity={fmt(metrics["uniformity"])}\n'
                 'Circle/angle panels: shared SVD projection for display only', fontsize=11)
    fig.savefig(output_dir / 'representation.png', dpi=180)
    fig.savefig(output_dir / 'representation.svg')
    plt.close(fig)


def write_report(output_dir, features, labels, sample_ids, metadata, align_alpha=2.0, uniform_t=2.0):
    z, y, ids = validate_features(features, labels, sample_ids)
    metrics, u, same_hist, different_hist = compute_metrics(z, y, align_alpha, uniform_t)
    basis, circle, valid, projection_counts = project_circle(u, y)
    metrics.update(projection_counts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'features.npz', features=z, labels=y, sample_ids=ids)
    np.save(output_dir / 'projection_basis.npy', basis)
    with (output_dir / 'sample_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'label'])
        writer.writerows(zip(ids, y))
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    metadata = dict(metadata)
    metadata.update({'dimension': int(z.shape[1]), 'align_alpha': align_alpha, 'uniform_t': uniform_t,
                     'metrics_space': 'raw exported feature dimensions, row L2-normalized',
                     'projection': 'shared uncentered SVD basis for display only; independent per report',
                     'circle_bandwidth': 0.10, 'angle_kappa': 8.0, 'figure_dpi': 180})
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    draw_figure(output_dir, metrics, same_hist, different_hist, circle, valid, y,
                metadata.get('model', 'Unknown model'), metadata.get('representation', 'unknown'), z.shape[1])
    return metrics
