"""Deterministic array regressions for selective, non-destructive alpha cleanup."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


UTILS = Path(__file__).resolve().parents[2] / 'common' / 'utils'
SPEC = importlib.util.spec_from_file_location('background_cleanup', UTILS / 'background_cleanup.py')
CLEANUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEANUP)


def make_part(alpha, origin=(0, 0), white=False):
    height, width = alpha.shape
    image = np.full((height, width, 4), 255 if white else 41, dtype=np.uint8)
    image[..., 3] = alpha
    x, y = origin
    return {'img': image, 'xyxy': [x, y, x + width, y + height],
            'depth': np.arange(height * width, dtype=np.float32).reshape(height, width),
            'depth_median': 0.375, 'rank': 7, 'tag': 'topwear'}


class BackgroundCleanupTests(unittest.TestCase):
    def test_leakage_weights_translucent_pixels_by_alpha_mass(self):
        alpha = np.full((10, 10), 100, dtype=np.uint8)
        part = make_part(alpha, origin=(3, 5))
        background = np.zeros((20, 20), dtype=bool)
        background[5, 3] = True
        self.assertEqual(CLEANUP.layer_leakage_ratio(part, background), 0.01)
        part['img'][0, 0, 3] = 1
        self.assertAlmostEqual(CLEANUP.layer_leakage_ratio(part, background), 1 / 9901)

    def test_white_foreground_translucent_details_and_hidden_completion_survive(self):
        alpha = np.array([[255, 173, 1, 0], [255, 255, 90, 7]], dtype=np.uint8)
        part = make_part(alpha, white=True)
        before = part['img'].copy()
        depth = part['depth']
        # Source says the last two columns are background, including a hidden
        # generated completion. This only selects the layer, never cuts it.
        background = np.zeros((4, 4), dtype=bool)
        background[:2, 2:] = True
        self.assertGreater(CLEANUP.layer_leakage_ratio(part, background), 0.01)
        probability = np.ones((4, 4), dtype=np.float32)
        probability[0, 0] = 0.0  # isolated actual artifact
        probability[0, 1] = 0.5  # exact preservation boundary
        result = CLEANUP.constrain_layer_alpha(part, probability, (4, 4))
        self.assertTrue(result['changed'])
        self.assertEqual(result['changed_pixels'], 1)
        self.assertEqual(part['img'][0, 0, 3], 0)
        np.testing.assert_array_equal(part['img'][..., 3][alpha != 255], alpha[alpha != 255])
        np.testing.assert_array_equal(part['img'][..., :3], before[..., :3])
        self.assertIs(part['depth'], depth)
        self.assertEqual(part['xyxy'], [0, 0, 4, 2])
        self.assertEqual(part['rank'], 7)
        self.assertEqual(part['depth_median'], 0.375)
        self.assertEqual(part['img'][1, 2, 3], 90)

    def test_nonzero_origin_uses_full_canvas_classifier_coordinates(self):
        part = make_part(np.full((2, 3), 255, dtype=np.uint8), origin=(4, 2))
        probability = np.ones((8, 8), dtype=np.float32)
        probability[2, 4] = 0.0
        probability[3, 6] = 0.25
        result = CLEANUP.constrain_layer_alpha(part, probability, (8, 8))
        np.testing.assert_array_equal(part['img'][..., 3], [[0, 255, 255], [255, 255, 127]])
        self.assertEqual(result['removed_alpha_mass'], 383)
        self.assertEqual(part['xyxy'], [4, 2, 7, 4])

    def test_minimum_size_box_beyond_numpy_crop_uses_actual_pixels(self):
        part = make_part(np.full((32, 1024), 255, dtype=np.uint8))
        part['xyxy'] = [0, 0, 1056, 32]
        box = part['xyxy']
        background = np.zeros((1024, 1024), dtype=bool)
        background[0, :] = True
        self.assertEqual(CLEANUP.layer_leakage_ratio(part, background), 1 / 32)
        rgb = CLEANUP.layer_rgb_on_white(part, (1024, 1024))
        np.testing.assert_array_equal(rgb[:32], part['img'][..., :3])
        self.assertTrue(np.all(rgb[32:] == 255))
        probability = np.ones((1024, 1024), dtype=np.float32)
        probability[0, :] = 0.0
        result = CLEANUP.constrain_layer_alpha(part, probability, (1024, 1024))
        self.assertEqual(result['changed_pixels'], 1024)
        self.assertTrue(np.all(part['img'][0, :, 3] == 0))
        self.assertTrue(np.all(part['img'][1:, :, 3] == 255))
        self.assertIs(part['xyxy'], box)
        self.assertEqual(part['xyxy'], [0, 0, 1056, 32])

    def test_entire_layer_deletion_is_rejected_without_modification(self):
        part = make_part(np.array([[0, 1], [100, 255]], dtype=np.uint8), origin=(3, 4))
        before = part['img'].copy()
        result = CLEANUP.constrain_layer_alpha(part, np.zeros((8, 8), dtype=np.float32), (8, 8))
        self.assertTrue(result['kept_unresolved'])
        self.assertFalse(result['changed'])
        self.assertEqual(result['removed_alpha_mass'], 0)
        np.testing.assert_array_equal(part['img'], before)

    def test_negative_origin_clips_canvas_not_original_layer_geometry(self):
        part = make_part(np.full((2, 3), 255, dtype=np.uint8), origin=(-1, 2))
        probability = np.ones((4, 4), dtype=np.float32)
        probability[2, 0] = 0.0
        CLEANUP.constrain_layer_alpha(part, probability, (4, 4))
        np.testing.assert_array_equal(part['img'][..., 3], [[255, 0, 255], [255, 255, 255]])
        self.assertEqual(part['xyxy'], [-1, 2, 2, 4])

    def test_white_composite_preserves_placement_and_removes_invisible_rgb(self):
        part = make_part(np.array([[255, 128, 0]], dtype=np.uint8), origin=(2, 3))
        part['img'][0, :, :3] = 0
        rgb = CLEANUP.layer_rgb_on_white(part, (8, 8))
        expected = np.full((8, 8, 3), 255, dtype=np.uint8)
        expected[3, 2] = 0
        expected[3, 3] = 127
        np.testing.assert_array_equal(rgb, expected)
        self.assertEqual(part['img'][0, 2, 0], 0)

    def test_nan_classifier_cannot_modify_original_alpha(self):
        part = make_part(np.full((2, 2), 255, dtype=np.uint8))
        before = part['img'].copy()
        probability = np.ones((4, 4), dtype=np.float32)
        probability[1, 1] = np.nan
        with self.assertRaises(ValueError):
            CLEANUP.constrain_layer_alpha(part, probability, (4, 4))
        np.testing.assert_array_equal(part['img'], before)


if __name__ == '__main__':
    unittest.main()
