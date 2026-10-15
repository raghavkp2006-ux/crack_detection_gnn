"""Pure mask processing and historical pitch metrics for the one-day experiment.

This module has no dataset discovery or test-data access. Distances/areas are
measured in native-image pixels. GT is used only by MetricSet, never processing.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import torch

S8 = np.ones((3, 3), dtype=np.uint8)
BINS = ["[0,50)", "[50,200)", "[200,inf)"]
OVERLAPS = [.3, .4, .5, .6, .7]
THRESHOLDS = sorted(set(round(.1 + .05 * i, 2) for i in range(17)) | {.36, .66})
GRIDS = dict(thresholds=THRESHOLDS, min_areas=[0, 5, 10, 20, 50],
             max_gaps=[0, 2, 4, 8, 12], alphas=[round(i / 10, 1) for i in range(11)],
             alignment_degrees=30, tangent_pixels=8, min_cr_gain=.01)


@dataclass(frozen=True)
class Recipe:
    low: float
    high: float
    min_area: int = 0
    max_gap: int = 0

    def json(self):
        return asdict(self)


def hysteresis(prob, low, high):
    if not 0 <= low <= high <= 1:
        raise ValueError("Expected 0 <= low <= high <= 1")
    support = prob >= np.float32(low)
    if low == high:
        return support
    labels, n = ndi.label(support, S8)
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(labels[prob >= np.float32(high)])] = True
    keep[0] = False
    return keep[labels]


def remove_small(mask, min_area):
    if min_area == 0:
        return mask.copy()
    labels, _ = ndi.label(mask, S8)
    keep = np.bincount(labels.ravel()) >= min_area
    keep[0] = False
    return keep[labels]


def _tangent(skel, endpoint, max_pixels=8):
    """Walk inwards until a branch, then return the outward unit tangent."""
    previous, current = None, tuple(endpoint)
    history = [current]
    for _ in range(max_pixels - 1):
        y, x = current
        neighbors = [(yy, xx) for yy in range(max(0, y-1), min(skel.shape[0], y+2))
                     for xx in range(max(0, x-1), min(skel.shape[1], x+2))
                     if (yy, xx) != current and (yy, xx) != previous and skel[yy, xx]]
        if len(neighbors) != 1:
            break
        previous, current = current, neighbors[0]
        history.append(current)
    vector = np.asarray(endpoint, float) - np.asarray(history[-1], float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def bridge_gaps(mask, max_gap):
    if max_gap == 0:
        return mask.copy()
    skel = skeletonize(mask)
    degree = ndi.convolve(skel.astype(np.uint8), S8, mode="constant") - skel
    endpoints = np.argwhere(skel & (degree == 1))
    if len(endpoints) < 2:
        return mask.copy()
    labels, _ = ndi.label(mask, S8)
    directions = [_tangent(skel, p) for p in endpoints]
    pairs = cKDTree(endpoints).query_pairs(max_gap)
    pairs = sorted(pairs, key=lambda p: (float(np.linalg.norm(endpoints[p[0]]-endpoints[p[1]])), p))
    result = mask.astype(np.uint8).copy()
    used = set()
    cosine = math.cos(math.radians(30))
    for i, j in pairs:
        a, b = endpoints[i], endpoints[j]
        if i in used or j in used or labels[tuple(a)] == labels[tuple(b)]:
            continue
        if directions[i] is None or directions[j] is None:
            continue
        connector = (b-a).astype(float)
        connector /= np.linalg.norm(connector)
        if directions[i] @ connector < cosine or directions[j] @ -connector < cosine:
            continue
        cv2.line(result, tuple(int(v) for v in a[::-1]), tuple(int(v) for v in b[::-1]), 1, 1)
        used.update((i, j))
    return result.astype(bool)


def process(prob, recipe):
    if isinstance(recipe, dict):
        recipe = Recipe(**recipe)
    return bridge_gaps(remove_small(hysteresis(prob, recipe.low, recipe.high),
                                   recipe.min_area), recipe.max_gap)


class MetricSet:
    """Cache GT once, preserving crack_metrics/rerun_pitch aggregation exactly."""
    def __init__(self, masks):
        self.gts = [m.astype(bool) for m in masks]
        self.skeletons = [skeletonize(m) for m in self.gts]
        self.parts = []
        for gt in self.gts:
            lab, _ = ndi.label(gt, S8)
            parts = []
            for cid, sl in enumerate(ndi.find_objects(lab), 1):
                component = lab[sl] == cid
                area = int(component.sum())
                parts.append((sl, component, area, 0 if area < 50 else 1 if area < 200 else 2))
            self.parts.append(parts)

    def score(self, predictions, detailed=False):
        counts = np.zeros((5 if detailed else 1, 3, 4), dtype=np.int64)
        thresholds = OVERLAPS if detailed else [.5]
        confusion = np.zeros(3, dtype=np.int64)
        cl_values, macro_values = [], [[] for _ in thresholds]
        for i, pred in enumerate(predictions):
            pred = np.asarray(pred, dtype=bool)
            gt, sg = self.gts[i], self.skeletons[i]
            if pred.shape != gt.shape:
                raise ValueError("Probability/GT shape mismatch")
            confusion += [(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()]
            if not pred.any() or not gt.any():
                cl_values.append(float(not pred.any() and not gt.any()))
            else:
                sp = skeletonize(pred)
                precision = float((sp & gt).sum()) / max(int(sp.sum()), 1)
                recall = float((sg & pred).sum()) / max(int(sg.sum()), 1)
                cl_values.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
            image_counts = np.zeros_like(counts)
            for sl, component, area, bucket in self.parts[i]:
                intersection = component & pred[sl]
                coverage = int(intersection.sum()) / area
                _, components = ndi.label(intersection, S8)
                for ti, threshold in enumerate(thresholds):
                    category = 2 if coverage < threshold else 1 if components == 1 else 3
                    image_counts[ti, bucket, 0] += 1
                    image_counts[ti, bucket, category] += 1
            counts += image_counts
            for ti in range(len(thresholds)):
                total = int(image_counts[ti, :, 0].sum())
                macro_values[ti].append(float(image_counts[ti, :, 1].sum()/total) if total else 0.)
        if len(cl_values) != len(self.gts):
            raise ValueError("Incomplete prediction set")
        index = 2 if detailed else 0
        total, preserved, missed, fragmented = map(int, counts[index].sum(0))
        tp, fp, fn = map(int, confusion)
        result = dict(cldice=float(np.mean(cl_values)), strict_cr=preserved/total if total else 0.,
                      pixel_f1=2*tp/max(2*tp+fp+fn, 1), n_total=total, n_preserved=preserved,
                      n_missed=missed, n_fragmented=fragmented, pixel_confusion=[tp, fp, fn])
        if detailed:
            result['size_bins'] = [dict(bin=b, total=int(c[0]), preserved=int(c[1]),
                                       missed=int(c[2]), fragmented=int(c[3]),
                                       strict_cr=float(c[1]/c[0]) if c[0] else 0.)
                                   for b, c in zip(BINS, counts[2])]
            result['overlap_sweep'] = [dict(overlap=t, strict_cr=float(c[:, 1].sum()/c[:, 0].sum())
                                          if c[:, 0].sum() else 0., macro=float(np.mean(macro_values[ti])))
                                       for ti, (t, c) in enumerate(zip(thresholds, counts))]
            result['cldice_per_image'] = cl_values
        return result


def rank(metrics):
    return metrics['strict_cr'], metrics['cldice'], metrics['pixel_f1']


def qualifies(candidate, incumbent, cldice_guard):
    return (candidate['strict_cr'] - incumbent['strict_cr'] >= .01 - 1e-12
            and candidate['cldice'] >= cldice_guard - 1e-12)


def skeleton_recall_loss(logits, skeleton_counts):
    denominator = skeleton_counts.sum()
    if denominator.item() == 0:
        return logits.sum() * 0
    return 1 - (torch.softmax(logits, 1)[:, 1] * skeleton_counts).sum()/denominator


def projected_run_seconds(epoch_seconds, max_epochs):
    if len(epoch_seconds) < 3:
        raise ValueError("Three measured epochs required")
    return sum(epoch_seconds[:3]) + max(epoch_seconds[1:3]) * max(0, max_epochs-3)
