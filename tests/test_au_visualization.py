"""Numerical and artifact checks for the independent A/U report."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from au_visualization import compute_metrics, project_circle, validate_features, write_report
from visualize_alignment_uniformity import _select_samples


class AlignmentUniformityTests(unittest.TestCase):
    def test_brute_anchor_and_pair_definitions(self):
        z = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
                      [0., 0., 1.], [0., 0., -1.]])
        y = np.array([0, 0, 0, 1, 1])
        result, u, _, _ = compute_metrics(z, y)
        anchor_means = [np.mean([np.linalg.norm(u[i] - u[j]) ** 2
                                 for j in range(len(y)) if i != j and y[i] == y[j]])
                        for i in range(len(y))]
        pair_terms = [math.exp(-2 * np.linalg.norm(u[i] - u[j]) ** 2)
                      for i in range(len(y)) for j in range(i + 1, len(y))]
        self.assertAlmostEqual(result['alignment'], float(np.mean(anchor_means)))
        self.assertAlmostEqual(result['uniformity'], math.log(np.mean(pair_terms)))
        self.assertEqual(result['same_class_pair_count'], 4)
        self.assertEqual(result['different_class_pair_count'], 6)
        self.assertNotAlmostEqual(result['alignment'], np.mean([np.linalg.norm(u[i] - u[j]) ** 2
                             for i in range(len(y)) for j in range(i + 1, len(y)) if y[i] == y[j]]))

    def test_scale_rotation_and_identical(self):
        z = np.array([[1., 2., 3.], [2., -1., 0.], [-1., 0., 2.], [0., 2., -1.]])
        y = np.array([0, 0, 1, 1])
        original = compute_metrics(z, y)[0]
        scaled = z.copy()
        scaled[0] *= 17
        rotated = z @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        for changed in (scaled, rotated):
            result = compute_metrics(changed, y)[0]
            self.assertAlmostEqual(result['alignment'], original['alignment'])
            self.assertAlmostEqual(result['uniformity'], original['uniformity'])
        same = compute_metrics(np.tile([1., 2.], (4, 1)), y)[0]
        self.assertAlmostEqual(same['alignment'], 0)
        self.assertAlmostEqual(same['uniformity'], 0)

    def test_degenerate_and_validation(self):
        z = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        y = np.array([0, 0, 1])
        result, u, _, _ = compute_metrics(z, y)
        self.assertEqual(result['zero_vector_count'], 2)
        self.assertIsNone(result['malicious_alignment'])
        self.assertEqual(result['valid_alignment_anchor_count'], 2)
        _, _, valid, counts = project_circle(u, y)
        self.assertEqual(valid.tolist(), [False, False, True])
        self.assertEqual(counts['projection_degenerate_count'], 2)
        self.assertIsNone(compute_metrics(z[:1], y[:1])[0]['uniformity'])
        self.assertIsNone(compute_metrics(z[2:], y[2:])[0]['alignment'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_features(z, y, np.array(['a', 'a', 'c']))
        with self.assertRaises(ValueError):
            validate_features(np.array([[np.nan, 0.]]), np.array([0]), np.array(['a']))
        with self.assertRaises(ValueError):
            validate_features(z, np.array([0, 0, 2]), np.array(['a', 'b', 'c']))

    def test_synthetic_artifacts_and_cache_roundtrip(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=(12, 5))
        y = np.array([0] * 6 + [1] * 6)
        ids = np.array([f'val/{"benign" if label == 0 else "mal"}/{i}.png'
                        for i, label in enumerate(y)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {'model': 'SYNTHETIC', 'representation': 'test_vectors',
                    'training_seed': None, 'sampling_seed': 42, 'split': 'synthetic'}
            metrics = write_report(root, z, y, ids, meta)
            for name in ('features.npz', 'representation.png', 'representation.svg',
                         'projection_basis.npy', 'metrics.json', 'metrics.csv',
                         'sample_manifest.csv', 'metadata.json'):
                self.assertGreater((root / name).stat().st_size, 0)
            self.assertEqual(json.loads((root / 'metrics.json').read_text())['sample_count'], len(z))
            with np.load(root / 'features.npz', allow_pickle=False) as cache:
                self.assertTrue(np.array_equal(cache['features'], z))
                self.assertTrue(np.array_equal(cache['sample_ids'], ids))
                self.assertAlmostEqual(compute_metrics(cache['features'], cache['labels'])[0]['uniformity'],
                                       metrics['uniformity'])

    def test_empty_class_and_zero_projection_render(self):
        z = np.zeros((3, 8), dtype=float)
        y = np.zeros(3, dtype=int)
        with tempfile.TemporaryDirectory() as directory:
            metrics = write_report(directory, z, y, np.array(['a', 'b', 'c']),
                                   {'model': 'SYNTHETIC', 'representation': 'zeros'})
            self.assertIsNone(metrics['malicious_alignment'])
            self.assertEqual(metrics['zero_vector_count'], 3)
            self.assertEqual(metrics['projection_degenerate_count'], 3)
            self.assertTrue((Path(directory) / 'representation.png').is_file())

    def test_wide_uncentered_projection(self):
        rng = np.random.default_rng(9)
        z = rng.normal(size=(5, 11))
        u = z / np.linalg.norm(z, axis=1, keepdims=True)
        basis, _, valid, _ = project_circle(u, np.array([0, 0, 1, 1, 1]))
        _, _, vt = np.linalg.svd(u, full_matrices=False)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
        np.testing.assert_allclose((u @ basis) @ (u @ basis).T,
                                   (u @ vt[:2].T) @ (u @ vt[:2].T).T, atol=1e-12)

    def test_pair_blocks_cover_every_sample_pair(self):
        rng = np.random.default_rng(11)
        z = rng.normal(size=(11, 6))
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        small, _, same_small, different_small = compute_metrics(z, y, pair_block_size=3)
        large, _, same_large, different_large = compute_metrics(z, y, pair_block_size=99)
        self.assertEqual(small['pair_count'], 55)
        self.assertEqual(int(same_small.sum() + different_small.sum()), 55)
        np.testing.assert_array_equal(same_small, same_large)
        np.testing.assert_array_equal(different_small, different_large)
        self.assertAlmostEqual(small['alignment'], large['alignment'])
        self.assertAlmostEqual(small['uniformity'], large['uniformity'])

    def test_default_selection_is_complete_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [(str(root / 'val' / 'benign' / f'{i}.png'), 0) for i in range(3)]
            paths += [(str(root / 'val' / 'mal' / f'{i}.png'), 1) for i in range(2)]

            class FakeDataset:
                samples = paths

                def __len__(self):
                    return len(self.samples)

            full = _select_samples(FakeDataset(), root, 'val', None, None, 42)
            self.assertEqual([index for index, _, _ in full], list(range(5)))
            self.assertEqual(len({sample_id for _, sample_id, _ in full}), 5)
            sampled = _select_samples(FakeDataset(), root, 'val', None, 2, 42)
            self.assertEqual(len(sampled), 2)


if __name__ == '__main__':
    unittest.main()
