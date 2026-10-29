"""Consumer-rendered alpha and depth roundtrips through the real PSD writer."""

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from psd_tools import PSDImage


SOURCE = Path(__file__).resolve().parents[2] / 'common' / 'utils' / 'io_utils.py'


def psd_writer():
    tree = ast.parse(SOURCE.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'save_psd')
    namespace = {'Image': Image, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['save_psd']


class PsdExportTests(unittest.TestCase):
    def export(self, parts, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layers.psd'
            psd_writer()(path, parts, 24, 24, **kwargs)
            return PSDImage.open(path)

    def test_partial_transparency_is_applied_once_after_reopening(self):
        values = np.array([0, 1, 15, 32, 80, 128, 200, 254, 255], dtype=np.uint8)
        rgba = np.full((4, len(values), 4), [230, 91, 147, 255], dtype=np.uint8)
        rgba[..., 3] = values
        doc = self.export([{'img': rgba, 'tag': 'fine-details', 'xyxy': [5, 7, 14, 11]}])
        np.testing.assert_array_equal(np.array(doc[0].topil().convert('RGBA')), rgba)
        rendered = np.array(doc.composite(force=True, ignore_preview=True).convert('RGBA'))
        expected = np.zeros((24, 24), dtype=np.uint8)
        expected[7:11, 5:14] = rgba[..., 3]
        np.testing.assert_array_equal(rendered[..., 3], expected)

    def test_overlapping_translucent_layers_preserve_source_over_coverage(self):
        bottom = np.full((5, 7, 4), [220, 80, 30, 80], dtype=np.uint8)
        top = np.full((5, 7, 4), [40, 170, 210, 128], dtype=np.uint8)
        doc = self.export([
            {'img': bottom, 'tag': 'bottom', 'xyxy': [3, 4, 10, 9]},
            {'img': top, 'tag': 'top', 'xyxy': [3, 4, 10, 9]},
        ])
        rendered = np.array(doc.composite(force=True, ignore_preview=True).convert('RGBA'))
        expected_alpha = round(128 + 80 * (1 - 128 / 255))
        np.testing.assert_allclose(rendered[4:9, 3:10, 3], expected_alpha, atol=1, rtol=0)
        outside = rendered[..., 3].copy()
        outside[4:9, 3:10] = 0
        self.assertTrue(np.all(outside == 0))

    def test_grayscale_depth_export_keeps_samples_and_origin(self):
        depth = np.arange(48, dtype=np.uint8).reshape(6, 8) * 5
        doc = self.export([{'depth': depth, 'tag': 'depth', 'xyxy': [4, 3, 12, 9]}],
                          mode='L', img_key='depth')
        self.assertEqual(doc[0].bbox, (4, 3, 12, 9))
        np.testing.assert_array_equal(np.array(doc[0].topil().convert('L')), depth)
        rendered = np.array(doc.composite(force=True, ignore_preview=True).convert('L'))
        np.testing.assert_array_equal(rendered[3:9, 4:12], depth)


if __name__ == '__main__':
    unittest.main()
