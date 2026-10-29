"""Conservative, per-layer alpha cleanup; source support only selects layers.

The source mask is never applied to generated pixels. A selected layer is
classified independently, including its hidden completions, on the full canvas.
"""

from contextlib import contextmanager
import gc
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_REPO = 'skytnt/anime-seg'
MODEL_REVISION = '493cb60893f47441b26ec4fb9a306bce9e342982'
MODEL_FILE = 'isnetis.ckpt'
MODEL_SHA256 = '2c8f6b9a77386c54dcdbf55b6c917108c4bdf4328abca9152c7bce5727b74d18'
LEAKAGE_THRESHOLD = 0.01


def _probability(value, shape=None):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or (shape is not None and value.shape != tuple(shape)):
        raise ValueError('Expected a probability map matching the image dimensions')
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError('Classifier probabilities must be finite and within [0, 1]')
    return value


def _frame_shape(frame_size):
    height, width = frame_size
    if height <= 0 or height != width or int(height) != height or int(width) != width:
        raise ValueError('Background cleanup requires a positive square (H, W) canvas')
    return int(height), int(width)


def project_source_probability(probability, frame_size):
    """Use the same centered square padding and resize as the inference input."""
    from .cv import center_square_pad_resize

    height, _ = _frame_shape(frame_size)
    probability = _probability(probability)
    return center_square_pad_resize(probability, height, pad_value=0)


def _layer_region(part, frame_size):
    height, width = _frame_shape(frame_size)
    image = part['img']
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('Layer images must be HWC uint8 RGBA arrays')
    ih, iw = image.shape[:2]
    box = part.get('xyxy', (0, 0, iw, ih))
    if len(box) != 4 or any(int(v) != v for v in box):
        raise ValueError('Layer xyxy must contain four integer coordinates')
    x1, y1 = map(int, box[:2])
    # Existing minimum-size crops can extend xyxy beyond the NumPy slice.
    # The stored origin and actual pixels determine support, not that extent.
    x2, y2 = x1 + iw, y1 + ih
    left, top, right, bottom = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if right <= left or bottom <= top:
        return image, None, None
    canvas = np.s_[top:bottom, left:right]
    local = np.s_[top - y1:bottom - y1, left - x1:right - x1]
    return image, canvas, local


def layer_leakage_ratio(part, background):
    """Fraction of original alpha mass on high-confidence source background."""
    background = np.asarray(background)
    if background.ndim != 2 or background.dtype != np.bool_:
        raise ValueError('Background support must be a two-dimensional boolean array')
    image, canvas, local = _layer_region(part, background.shape)
    total = int(image[..., 3].sum(dtype=np.uint64))
    if total == 0 or canvas is None:
        return 0.0
    leaked = int(image[..., 3][local][background[canvas]].sum(dtype=np.uint64))
    return float(leaked / total)


def layer_rgb_on_white(part, frame_size):
    """Composite a single layer onto a full white RGB classification canvas."""
    image, canvas, local = _layer_region(part, frame_size)
    rgb = np.full((*_frame_shape(frame_size), 3), 255, dtype=np.uint8)
    if canvas is not None:
        pixels = image[local]
        alpha = pixels[..., 3:4].astype(np.float32) / 255.0
        rgb[canvas] = np.rint(pixels[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return rgb


def constrain_layer_alpha(part, probability, frame_size):
    """Only reduce alpha; reject a proposed whole-layer deletion atomically.

    Pixels whose independent layer classifier gives p >= .5 retain their exact
    original alpha, including translucent white edges and hidden completions.
    """
    probability = _probability(probability, _frame_shape(frame_size))
    image, canvas, local = _layer_region(part, frame_size)
    result = {'changed': False, 'kept_unresolved': False, 'changed_pixels': 0,
              'removed_alpha_mass': 0}
    if canvas is None:
        return result
    original = image[..., 3]
    bounded = np.minimum(original[local], np.minimum(2.0 * probability[canvas], 1.0) * 255.0).astype(np.uint8)
    removed = int(original[local].sum(dtype=np.uint64)) - int(bounded.sum(dtype=np.uint64))
    if removed == 0:
        return result
    if removed == int(original.sum(dtype=np.uint64)):
        result['kept_unresolved'] = True
        return result
    result.update(changed=True, changed_pixels=int(np.count_nonzero(original[local] != bounded)),
                  removed_alpha_mass=removed)
    original[local] = bounded
    return result


@contextmanager
def _cpu_threads():
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous, 8))
        yield
    finally:
        torch.set_num_threads(previous)


def _download_weights():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                           revision=MODEL_REVISION, token=False)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError('Background cleanup checkpoint SHA256 mismatch')
    return path


def _load_model(path):
    import torch
    from annotators.animeinsseg.animeseg_refine_model import ISNetDIS

    state = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError('Background cleanup checkpoint must contain a state dict')
    state = state.get('state_dict', state)
    if not isinstance(state, dict):
        raise ValueError('Background cleanup checkpoint state_dict must be a mapping')
    state = {key[4:]: value for key, value in state.items() if key.startswith('net.')}
    if not state:
        raise ValueError('Background cleanup checkpoint contains no net.* weights')
    # Training/GT branches are not part of ISNetDIS. Every target net.* weight
    # must still exactly match, with no missing or unexpected target keys.
    with torch.device('meta'):
        model = ISNetDIS()
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device='cpu', dtype=torch.float32).eval()


def _classify(model, rgb):
    import torch
    from annotators.animeinsseg.skytnt_segmentation import apply_segmentation

    with torch.inference_mode():
        probability = apply_segmentation(rgb, use_amp=False, s=1024, model=model)
    return _probability(probability, rgb.shape[:2])


class BackgroundCleanup:
    """Prepare before diffusion; load a CPU classifier only for selected layers."""

    def __init__(self, image_path, progress_callback=None):
        self.image_path = Path(image_path)
        self.progress_callback = progress_callback
        self._weights_path = None
        self._source_probability = None
        self.report = {'enabled': True, 'source_mode': None, 'changed': False,
                       'kept_unresolved': [], 'layers': []}

    def _progress(self, stage):
        if self.progress_callback is not None:
            self.progress_callback(stage)

    def prepare(self):
        self._progress('background_cleanup_prepare')
        self._source_probability = None
        model = None
        try:
            with _cpu_threads():
                self._weights_path = _download_weights()
                with Image.open(self.image_path) as source:
                    rgba = np.array(source.convert('RGBA'))
                if np.any(rgba[..., 3] < 255):
                    self._source_probability = rgba[..., 3].astype(np.float32) / 255.0
                    self.report['source_mode'] = 'alpha'
                else:
                    model = _load_model(self._weights_path)
                    self._source_probability = _classify(model, rgba[..., :3])
                    self.report['source_mode'] = 'isnet'
        finally:
            del model
            gc.collect()
        return self

    def refine_parts(self, parts, frame_size):
        if self._source_probability is None or self._weights_path is None:
            raise RuntimeError('BackgroundCleanup.prepare() must finish before refining layers')
        source = project_source_probability(self._source_probability, frame_size)
        background = np.rint(source * 255.0) == 0
        self.report.update(changed=False, kept_unresolved=[], layers=[])
        entries = list(parts.items()) if isinstance(parts, dict) else list(enumerate(parts))
        selected = []
        for index, (key, part) in enumerate(entries):
            ratio = layer_leakage_ratio(part, background)
            record = {'index': int(index), 'tag': str(part.get('tag', key)),
                      'leakage_ratio': float(ratio), 'selected': bool(ratio >= LEAKAGE_THRESHOLD),
                      'changed': False, 'kept_unresolved': False, 'changed_pixels': 0,
                      'removed_alpha_mass': 0}
            self.report['layers'].append(record)
            if record['selected']:
                selected.append((part, record))
        if not selected:
            return parts
        self._progress('background_cleanup_layers')
        model = None
        try:
            with _cpu_threads():
                # Recheck the cached bytes on reload; prepare retains no NN state.
                self._weights_path = _download_weights()
                model = _load_model(self._weights_path)
                for part, record in selected:
                    probability = _classify(model, layer_rgb_on_white(part, frame_size))
                    record.update(constrain_layer_alpha(part, probability, frame_size))
                    if record['kept_unresolved']:
                        self.report['kept_unresolved'].append(record['tag'])
                    self.report['changed'] = bool(self.report['changed'] or record['changed'])
        finally:
            del model
            gc.collect()
        return parts
