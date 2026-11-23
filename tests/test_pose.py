"""Fast CPU tests: SimpleBaseline shapes, crop geometry, heatmap encode/decode, flip test, COCO OKS AP."""
import contextlib
import io

import cv2
import numpy as np
import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import FLIP_PAIRS, CocoKeypoints, affine, apply, box_to_center_scale
from evaluate import decode, flip_back
from model import SimpleBaseline


@pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
def test_simplebaseline_output_shape(backbone):
    m = SimpleBaseline(backbone, pretrained=False).eval()          # no ImageNet download
    with torch.no_grad():
        y = m(torch.randn(2, 3, 256, 192))
    assert y.shape == (2, 17, 64, 48)


def test_box_to_center_scale_aspect_and_padding():
    c, s = box_to_center_scale(10, 20, 30, 100)                     # tall box -> widen to 3:4
    np.testing.assert_allclose(c, [25, 70])
    np.testing.assert_allclose(s, [75 * 1.25, 100 * 1.25])
    c, s = box_to_center_scale(0, 0, 120, 40)                       # wide box -> heighten
    np.testing.assert_allclose(s / 1.25, [120, 160])


def test_affine_inverse_round_trip():
    c, s = np.array([200.0, 150.0]), np.array([90.0, 120.0])
    M, Minv = affine(c, s, 0, (192, 256)), affine(c, s, 0, (192, 256), inv=True)
    np.testing.assert_allclose(apply(M, c[None]), [[96, 128]], atol=1e-4)  # box centre -> crop centre
    pts = np.array([[160.0, 100.0], [230.0, 200.0], [200.0, 150.0]])
    np.testing.assert_allclose(apply(Minv, apply(M, pts)), pts, atol=1e-3)
    R = affine(c, s, 30, (192, 256))                                # rotation keeps the centre fixed
    np.testing.assert_allclose(apply(R, c[None]), [[96, 128]], atol=1e-4)


def _fake_dataset(tmp_path, kpts):
    cv2.imwrite(str(tmp_path / "img.jpg"), np.zeros((480, 640, 3), np.uint8))
    ds = object.__new__(CocoKeypoints)                              # skip the annotation file
    ds.split, ds.train, ds.img_dir = "val", False, str(tmp_path)
    ds.in_wh, ds.hm_wh, ds.sigma = (192, 256), (48, 64), 2.0
    c, s = box_to_center_scale(220, 60, 180, 360)
    ds.items = [{"img_id": 1, "file": "img.jpg", "center": c, "size": s, "kpts": kpts, "ann_id": 1, "area": 1.0}]
    return ds


def test_heatmap_encode_decode_round_trip(tmp_path):
    rng = np.random.RandomState(0)
    k = np.zeros((17, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = rng.uniform(240, 380, 17), rng.uniform(80, 400, 17), 2
    k[3, 2] = 0                                                     # one unlabelled joint
    ds = _fake_dataset(tmp_path, k)
    x, target, vis, meta = ds[0]
    assert x.shape == (3, 256, 192) and target.shape == (17, 64, 48)
    assert vis[3] == 0 and target[3].abs().sum() == 0 and vis.sum() == 16
    coords, maxval = decode(target[None].numpy())
    xy = apply(affine(meta["center"], meta["size"], 0, ds.in_wh, inv=True), coords[0] * 4.0)
    err = np.linalg.norm(xy - k[:, :2], axis=1)[vis.numpy() > 0]
    # quarter-pixel refinement on a 4x-downsampled heatmap: error below ~1 heatmap pixel (4 image px)
    assert err.max() < 4.0 and err.mean() < 2.5
    assert np.all(maxval[0][vis.numpy() > 0] > 0.5)


def test_flip_back_undoes_horizontal_flip():
    hm = torch.rand(1, 17, 64, 48)
    swap = list(range(17))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    flipped = hm.flip(3)[:, swap]                                   # what a flip-equivariant net would output
    back = flip_back(flipped.clone())
    torch.testing.assert_close(back[..., 1:], hm[..., :-1])         # identical up to the 1-px shift
    torch.testing.assert_close(back[..., 0], back[..., 1])            # border column is replicated


def _coco_eval(gt_kpts, dt_kpts):
    img = {"id": 1, "width": 640, "height": 480}
    ann = {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 200 * 300,
           "bbox": [200, 80, 200, 300], "num_keypoints": 17, "keypoints": gt_kpts.reshape(-1).tolist()}
    gt = COCO()
    gt.dataset = {"images": [img], "annotations": [ann], "categories": [{"id": 1, "name": "person"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes([{"image_id": 1, "category_id": 1, "keypoints": dt_kpts.reshape(-1).tolist(), "score": 0.9}])
        E = COCOeval(gt, dt, "keypoints")
        E.params.useSegm = None
        E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]


def test_coco_oks_ap_perfect_vs_perturbed():
    rng = np.random.RandomState(1)
    gt = np.zeros((17, 3)); gt[:, 0], gt[:, 1], gt[:, 2] = rng.uniform(220, 380, 17), rng.uniform(100, 360, 17), 2
    dt = gt.copy(); dt[:, 2] = 1.0
    assert _coco_eval(gt, dt) == pytest.approx(1.0)
    far = dt.copy(); far[:, :2] += rng.normal(0, 25, (17, 2))     # large errors -> low OKS
    assert _coco_eval(gt, far) < 0.5
