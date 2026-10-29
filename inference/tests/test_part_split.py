"""CPU-only, synthetic regressions for lossless PSD component splitting."""

import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

UTILS = Path(__file__).resolve().parents[2] / 'common' / 'utils'
SPEC = importlib.util.spec_from_file_location('part_split', UTILS / 'part_split.py')
SPLIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLIT)


def make_part(alpha, origin=(37, 53)):
    height, width = alpha.shape
    yy, xx = np.indices(alpha.shape)
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[..., 0] = xx * 3
    img[..., 1] = yy * 5
    img[..., 2] = 123
    img[..., 3] = alpha
    return {'img': img, 'depth': (xx + yy).astype(np.float32) / (height + width),
            'mask': alpha > 10, 'xyxy': [origin[0], origin[1],
                                      origin[0] + width, origin[1] + height],
            'tag': 'ears'}


def inference_functions(**dependencies):
    # Execute the real CPU functions without importing CUDA/model dependencies.
    source = ast.parse((UTILS / 'inference_utils.py').read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {'save_part', '_split_psd_parts', '_apply_face_depth_order',
                               'further_extr', 'dump_parts_psd'}]
    namespace = {'np': np, 'cv2': cv2, 'os': os, 'osp': os.path,
                 'eyes_split': SPLIT.eyes_split, 'tag_lr_split': SPLIT.tag_lr_split,
                 'order_eye_layers': SPLIT.order_eye_layers}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(UTILS / 'inference_utils.py'), 'exec'), namespace)
    namespace.update(dependencies)
    return namespace


class PartSplitTests(unittest.TestCase):
    def assert_conserved(self, source, children):
        height, width = source['img'].shape[:2]
        total_alpha = np.zeros((height, width), dtype=np.uint16)
        owners = np.zeros((height, width), dtype=np.uint8)
        sx, sy = source['xyxy'][:2]
        for child in children.values():
            x1, y1, x2, y2 = child['xyxy']
            x1, x2, y1, y2 = x1 - sx, x2 - sx, y1 - sy, y2 - sy
            self.assertEqual(child['img'].shape[:2], (y2 - y1, x2 - x1))
            region = np.s_[y1:y2, x1:x2]
            visible = child['img'][..., -1] > 0
            total_alpha[region] += child['img'][..., -1]
            owners[region] += visible
            np.testing.assert_array_equal(child['img'][..., :3][visible],
                                          source['img'][region][..., :3][visible])
            np.testing.assert_array_equal(child['depth'][visible],
                                          source['depth'][region][visible])
        np.testing.assert_array_equal(total_alpha, source['img'][..., -1])
        np.testing.assert_array_equal(owners, source['img'][..., -1] > 0)

    def test_foreground_larger_than_background_is_not_dropped(self):
        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[:, :13] = 255
        alpha[2:18, 15:19] = 173
        source = make_part(alpha)
        children = SPLIT.part_lr_split('ears', source)
        self.assertEqual(set(children), {'ears-r', 'ears-l'})
        self.assert_conserved(source, children)
        self.assertEqual(children['ears-r']['xyxy'], [37, 53, 50, 73])
        self.assertEqual(children['ears-l']['xyxy'], [52, 55, 56, 71])

    def test_extra_islands_noise_and_faint_alpha_are_owned_once(self):
        alpha = np.zeros((30, 40), dtype=np.uint8)
        alpha[5:15, 4:10] = 255
        alpha[5:15, 28:34] = 220
        alpha[19:22, 7:10] = 180
        alpha[19:22, 29:32] = 180
        alpha[26, 39] = 255
        alpha[4, 3:11] = 3
        alpha[4, 27:35] = 7
        source = make_part(alpha, origin=(-9, 81))
        before = source['img'].copy()
        before_depth = source['depth'].copy()
        children = SPLIT.part_lr_split('ears', source)
        self.assert_conserved(source, children)
        np.testing.assert_array_equal(source['img'], before)
        np.testing.assert_array_equal(source['depth'], before_depth)
        self.assertLess(children['ears-r']['xyxy'][0], children['ears-l']['xyxy'][0])

    def test_close_disconnected_diagonals_do_not_duplicate_alpha(self):
        alpha = np.zeros((20, 20), dtype=np.uint8)
        for y in range(2, 16):
            alpha[y, y] = 255
            alpha[y, y + 3] = 191
        source = make_part(alpha)
        self.assert_conserved(source, SPLIT.part_lr_split('irides', source))

    def test_touching_counterparts_and_empty_parts_remain_unsplit(self):
        for empty in (False, True):
            alpha = np.zeros((15, 25), dtype=np.uint8)
            if not empty:
                alpha[3:12, 2:8] = 255
                alpha[3:12, 17:23] = 255
                alpha[7, 8:17] = 255
            source = make_part(alpha)
            result = SPLIT.part_lr_split('ears', source)
            self.assertEqual(list(result), ['ears'])
            self.assertIs(result['ears'], source)

    def test_legacy_two_or_three_components_do_not_vanish(self):
        for count in (2, 3):
            alpha = np.zeros((20, 35), dtype=np.uint8)
            for index in range(count):
                alpha[4:10, index * 10 + 2:index * 10 + 7] = 255
            source = make_part(alpha)
            result = SPLIT.eyes_split(source)
            self.assertIs(result['eyes'], source)
            self.assert_conserved(source, result)

    def test_legacy_eyes_keep_extra_islands_and_anatomical_names(self):
        alpha = np.zeros((30, 40), dtype=np.uint8)
        alpha[12:20, 3:10] = 255
        alpha[12:20, 29:36] = 255
        alpha[4:6, 3:10] = 200
        alpha[4:6, 29:36] = 200
        alpha[25, 5] = 255
        source = make_part(alpha)
        children = SPLIT.eyes_split(source)
        self.assertEqual(set(children), {'eyer', 'eyel', 'browr', 'browl'})
        self.assert_conserved(source, children)
        self.assertLess(children['eyer']['xyxy'][0], children['eyel']['xyxy'][0])
        self.assertLess(children['browr']['xyxy'][0], children['browl']['xyxy'][0])

    def test_children_can_be_split_again_with_local_masks(self):
        alpha = np.zeros((30, 40), dtype=np.uint8)
        alpha[3:10, 3:9] = 255
        alpha[3:10, 28:34] = 255
        alpha[20:23, 5:8] = 200
        source = make_part(alpha)
        child = SPLIT.part_lr_split('ears', source)['ears-r']
        self.assertEqual(child['mask'].shape, child['img'].shape[:2])
        grandchildren = SPLIT.part_lr_split('detail', child)
        self.assert_conserved(child, grandchildren)

    def test_final_save_crop_preserves_faint_alpha_and_origin(self):
        alpha = np.zeros((12, 14), dtype=np.uint8)
        alpha[4:8, 5:9] = 255
        alpha[1, 2] = 2
        source = make_part(alpha)
        saved = inference_functions()['save_part'](
            'ears', '', dict(source), save_to_disk=False)
        self.assertEqual(saved['xyxy'], [39, 54, 46, 61])
        self.assertEqual(int(saved['img'][..., -1].sum()), int(alpha.sum()))

    def test_disabled_split_preserves_legacy_eyes_at_export_boundary(self):
        alpha = np.zeros((25, 40), dtype=np.uint8)
        for x, y in ((2, 3), (25, 3), (2, 15), (25, 15)):
            alpha[y:y + 4, x:x + 6] = 255
        source = make_part(alpha)
        source['tag'] = 'eyes'
        exported = Mock()
        fullpage = np.zeros((100, 100, 4), dtype=np.uint8)
        functions = inference_functions(
            load_parts=Mock(return_value=(fullpage, {}, [source])),
            dump_parts_psd=exported)
        with tempfile.TemporaryDirectory() as directory:
            functions['further_extr'](directory, rotate=False,
                                      save_to_psd=True, tblr_split=False)
        parts = exported.call_args.args[0]
        self.assertEqual(set(parts), {'eyes'})
        self.assertEqual(int(parts['eyes']['img'][..., -1].sum()), int(alpha.sum()))

    def test_split_inherits_parent_depth_including_explicit_rank(self):
        alpha = np.zeros((12, 20), dtype=np.uint8)
        alpha[2:10, 2:7] = 255
        alpha[2:10, 13:18] = 255
        source = make_part(alpha)
        source['depth'][:, :10] = 0.9
        source['depth'][:, 10:] = 0.1
        for explicit in (False, True):
            expected = 0.37 if explicit else np.median(source['depth'][alpha > 10])
            if explicit:
                source['depth_median'] = expected
            children = SPLIT.part_lr_split('irides', source)
            for tag, child in children.items():
                self.assertEqual(child['depth_median'], expected)
                saved = inference_functions()['save_part'](
                    tag, '', dict(child), save_to_disk=False)
                self.assertEqual(saved['depth_median'], expected)

    def test_equal_depth_split_keeps_original_layer_slot(self):
        alpha = np.zeros((12, 20), dtype=np.uint8)
        alpha[2:10, 2:7] = 255
        alpha[2:10, 13:18] = 255
        source = make_part(alpha)
        parts = {'back': object(), 'irides': source, 'front': object()}
        SPLIT.tag_lr_split('irides', parts)
        self.assertEqual(list(parts), ['back', 'irides-r', 'irides-l', 'front'])

    def test_split_export_preserves_both_pupils_and_unsplit_composite(self):
        alpha = np.zeros((12, 20), dtype=np.uint8)
        alpha[2:10, 2:7] = 255
        alpha[2:10, 13:18] = 255
        composites = []
        for enabled in (False, True):
            irides = make_part(alpha, origin=(0, 0))
            irides['tag'] = 'irides'
            irides['img'][..., :3] = [0, 100, 0]
            irides['depth'][:, :10] = 0.9
            irides['depth'][:, 10:] = 0.1
            white = make_part(alpha, origin=(0, 0))
            white['tag'] = 'eyewhite'
            white['img'][..., :3] = 255
            white['depth'][:] = 0.6
            layers = []

            def capture(_path, parts, _height, _width, **kwargs):
                if kwargs.get('mode') != 'L':
                    layers.extend(dict(part) for part in parts)

            functions = inference_functions(
                load_parts=Mock(return_value=(np.zeros((12, 20, 4), dtype=np.uint8),
                                               {}, [white, irides])),
                save_psd=capture, dict2json=Mock())
            with tempfile.TemporaryDirectory() as directory:
                functions['further_extr'](directory, rotate=False,
                                          save_to_psd=True, tblr_split=enabled)
            composite = np.zeros((12, 20, 3), dtype=np.uint8)
            for part in layers:
                x1, y1, x2, y2 = part['xyxy']
                visible = part['img'][..., -1] > 0
                composite[y1:y2, x1:x2][visible] = part['img'][..., :3][visible]
            composites.append(composite)
            np.testing.assert_array_equal(composite[alpha > 0],
                                          np.tile([0, 100, 0], (np.count_nonzero(alpha), 1)))
        np.testing.assert_array_equal(composites[0], composites[1])

    def test_real_psd_keeps_pupils_visible_despite_inverted_or_tied_depth(self):
        # The depth relations reproduce both captured failure modes, while all
        # images are synthetic and no private inputs are needed by this test.
        io_source = ast.parse((UTILS / 'io_utils.py').read_text())
        save_node = next(node for node in io_source.body
                         if isinstance(node, ast.FunctionDef) and node.name == 'save_psd')
        io_namespace = {'Image': Image, 'np': np}
        exec(compile(ast.Module(body=[save_node], type_ignores=[]),
                     str(UTILS / 'io_utils.py'), 'exec'), io_namespace)
        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[3:9, 2:7] = 255
        alpha[3:9, 13:18] = 255
        for iris_depth, white_depth in ((0.415686, 0.388235), (0.411765, 0.411765)):
            composites = []
            for enabled in (False, True):
                with self.subTest(split=enabled, depths=(iris_depth, white_depth)):
                    parts = {}
                    for tag, color, depth in (
                            ('irides', [0, 130, 110], iris_depth),
                            ('eyewhite', [255, 255, 255], white_depth),
                            ('eyelash', [10, 10, 10], 0.8),
                            ('eyebrow', [80, 40, 20], 0.9)):
                        local_alpha = alpha.copy()
                        if tag == 'eyelash':
                            local_alpha[4:] = 0
                        elif tag == 'eyebrow':
                            local_alpha[:] = 0
                            local_alpha[1, 2:7] = 255
                            local_alpha[1, 13:18] = 255
                        part = make_part(local_alpha, origin=(0, 0))
                        part['tag'] = tag
                        part['img'][..., :3] = color
                        part['depth'][:] = depth
                        parts[tag] = part
                    # A non-eye layer deliberately lies between eye depth values.
                    nose_alpha = np.zeros_like(alpha)
                    nose_alpha[10, 10] = 255
                    nose = make_part(nose_alpha, origin=(0, 0))
                    nose['tag'] = 'nose'
                    nose['depth'][:] = 0.4
                    parts['nose'] = nose
                    if enabled:
                        for tag in ('eyewhite', 'irides', 'eyelash', 'eyebrow'):
                            SPLIT.tag_lr_split(tag, parts)
                    functions = inference_functions(
                        save_psd=io_namespace['save_psd'], dict2json=Mock())
                    for tag, part in parts.items():
                        functions['save_part'](tag, '', part, save_to_disk=False)
                    depth_sorted = sorted(parts.values(),
                                          key=lambda part: part['depth_median'], reverse=True)
                    nose_slot = next(index for index, part in enumerate(depth_sorted)
                                     if part['tag'] == 'nose')
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'synthetic.psd'
                        functions['dump_parts_psd'](parts, alpha.shape, str(path))
                        psd = PSDImage.open(path)
                        composite = np.array(psd.composite())
                        self.assertEqual(list(psd)[nose_slot].name, 'nose')
                        # Consumer-rendered opaque pupil centers must remain green.
                        np.testing.assert_array_equal(composite[6, 4, :3], [0, 130, 110])
                        np.testing.assert_array_equal(composite[6, 15, :3], [0, 130, 110])
                        # Eyelashes must stay above the irides, not just whites.
                        np.testing.assert_array_equal(composite[3, 4, :3], [10, 10, 10])
                        for tag in ('irides', 'eyewhite'):
                            visible_alpha = sum(
                                int(np.asarray(layer.topil())[..., -1].sum())
                                for layer in psd
                                if layer.name == tag or layer.name.startswith(tag + '-'))
                            self.assertEqual(visible_alpha, int(alpha.sum()))
                        composites.append(composite)
            np.testing.assert_array_equal(composites[0], composites[1])

if __name__ == '__main__':
    unittest.main()
