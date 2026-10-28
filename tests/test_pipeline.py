"""Quick reproducibility checks; run using Python's built-in unittest."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import synthetic_scene, binary_metrics, segment

class TestSegmentation(unittest.TestCase):
    def test_phantom_is_deterministic(self):
        a, mask_a = synthetic_scene(seed=123, condition='clean')
        b, mask_b = synthetic_scene(seed=123, condition='clean')
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(mask_a, mask_b))

    def test_perfect_prediction_metrics(self):
        _, ground_truth = synthetic_scene(seed=123, condition='clean')
        metrics = binary_metrics(ground_truth, ground_truth)
        self.assertAlmostEqual(metrics['dice'], 1.0)
        self.assertAlmostEqual(metrics['iou'], 1.0)

    def test_all_methods_generate_masks(self):
        image, ground_truth = synthetic_scene(seed=123, condition='noisy')
        for method in ('Otsu', 'KMeans-2', 'KMeans-3'):
            predicted, runtime = segment(image, method)
            self.assertEqual(predicted.shape, ground_truth.shape)
            self.assertEqual(predicted.dtype, np.dtype('bool'))
            self.assertGreater(runtime, 0)
            self.assertGreater(binary_metrics(predicted, ground_truth)['dice'], 0.7)

if __name__ == '__main__':
    unittest.main()
