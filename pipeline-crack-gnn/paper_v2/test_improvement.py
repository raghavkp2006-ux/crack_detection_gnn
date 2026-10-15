"""Synthetic verification; never loads real test images, labels or probabilities."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from scipy import ndimage as ndi
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from improvement_core import (MetricSet, Recipe, S8, bridge_gaps, hysteresis,
                              process, qualifies, remove_small, skeleton_recall_loss,
                              projected_run_seconds)
from improve_m2_m4 import BASE, Experiment, BudgetExceeded, save, load_config
from torch_geometric.data import Data
import crack_metrics as cm


class ImprovementTests(unittest.TestCase):
    def test_hysteresis_seeded_components_and_diagonals(self):
        p=np.zeros((8,8),np.float32)
        p[1,1]=.9; p[2,2]=.4; p[3,3]=.4; p[6,6]=.4
        result=hysteresis(p,.3,.8)
        self.assertEqual(int(result.sum()),3)
        self.assertFalse(result[6,6])
        np.testing.assert_array_equal(hysteresis(p,.4,.4),p>=np.float32(.4))
        with self.assertRaises(ValueError): hysteresis(p,.8,.3)

    def test_component_boundary(self):
        mask=np.zeros((10,10),bool); mask[1,1:6]=True; mask[7,1:5]=True
        result=remove_small(mask,5)
        self.assertEqual(int(result.sum()),5)
        np.testing.assert_array_equal(remove_small(mask,0),mask)

    def test_bridge_aligned_and_distance(self):
        mask=np.zeros((25,30),bool); mask[12,2:10]=True; mask[12,14:24]=True
        np.testing.assert_array_equal(bridge_gaps(mask,4),mask)
        result=bridge_gaps(mask,5)
        self.assertTrue(result[12,9:15].all())
        self.assertEqual(ndi.label(result,S8)[1],1)
        np.testing.assert_array_equal(bridge_gaps(mask,0),mask)

    def test_bridge_rejects_misaligned_endpoints(self):
        mask=np.zeros((25,30),bool); mask[12,2:10]=True; mask[12:22,14]=True
        np.testing.assert_array_equal(bridge_gaps(mask,5),mask)

    def test_disabled_pipeline_is_exact_threshold(self):
        p=np.random.RandomState(42).rand(30,30).astype(np.float32)
        np.testing.assert_array_equal(process(p,Recipe(.66,.66)),p>=np.float32(.66))

    def test_metrics_match_reference_random_and_empty(self):
        rng=np.random.RandomState(42)
        masks=[rng.rand(20,30)>.65,np.zeros((20,30),bool),np.ones((20,30),bool)]
        predictions=[rng.rand(20,30)>.6,np.zeros((20,30),bool),np.zeros((20,30),bool)]
        score=MetricSet(masks).score(predictions,True)
        self.assertAlmostEqual(score['cldice'],float(np.mean([cm.cl_dice(p,g) for p,g in zip(predictions,masks)])))
        reference=[cm.connectivity_ratio_breakdown(p,g,.5) for p,g in zip(predictions,masks)]
        self.assertEqual(score['n_total'],sum(r['n_total'] for r in reference))
        self.assertEqual(score['n_preserved'],sum(r['n_preserved'] for r in reference))

    def test_strict_intersection_rejects_outside_bridge(self):
        gt=np.zeros((25,30),bool); gt[12,3:23]=True
        pred=np.zeros_like(gt); pred[12,3:10]=True; pred[12,16:23]=True; pred[11,3:23]=True
        self.assertEqual(ndi.label(pred,S8)[1],1)
        score=MetricSet([gt]).score([pred])
        self.assertEqual(score['strict_cr'],0.)
        self.assertEqual(score['n_fragmented'],1)

    def test_exact_bin_boundaries(self):
        masks=[]
        for area in (49,50,199,200):
            mask=np.zeros((3,202),bool); mask[1,1:area+1]=True; masks.append(mask)
        score=MetricSet(masks).score(masks,True)
        self.assertEqual([b['total'] for b in score['size_bins']],[1,2,1])
        self.assertEqual([b['bin'] for b in score['size_bins']],['[0,50)','[50,200)','[200,inf)'])

    def test_one_percentage_point_and_guard(self):
        incumbent=dict(strict_cr=.3,cldice=.7,pixel_f1=.7)
        candidate=dict(strict_cr=.31,cldice=.7,pixel_f1=.6)
        self.assertTrue(qualifies(candidate,incumbent,.7))
        self.assertFalse(qualifies(dict(candidate,strict_cr=.30999),incumbent,.7))
        self.assertFalse(qualifies(dict(candidate,cldice=.69999),incumbent,.7))

    def test_skeleton_gradients_and_empty(self):
        logits=torch.tensor([[.1,.3],[.4,.2]],requires_grad=True)
        loss=skeleton_recall_loss(logits,torch.tensor([2.,0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0].abs().sum()),0.)
        empty=skeleton_recall_loss(logits,torch.zeros(2))
        self.assertEqual(float(empty.detach()),0.)

    def test_timing_gate_projection(self):
        self.assertEqual(projected_run_seconds([12.,10.,11.],150),33+147*11)
        self.assertGreater(projected_run_seconds([20.,13.,14.],150),1800)
        with self.assertRaises(ValueError): projected_run_seconds([1.],150)

    def test_output_protection_budget_and_test_access(self):
        with tempfile.TemporaryDirectory(prefix='improve_m2_m4_test_',dir=BASE/'results/paper_v2') as directory:
            path=Path(directory); experiment=Experiment(path)
            save(path/'sentinel.json',dict(unchanged=True))
            with self.assertRaises(FileExistsError): save(path/'sentinel.json',dict(unchanged=False))
            self.assertEqual(json.loads((path/'sentinel.json').read_text()),dict(unchanged=True))
            with self.assertRaises(RuntimeError): experiment.paths('test','synthetic',True)
            experiment.step_name='train'; experiment.deadline=time.time()+5
            with self.assertRaises(BudgetExceeded): experiment.check(estimate=10)
            experiment.check(estimate=1)

    def test_training_only_three_epoch_smoke(self):
        with tempfile.TemporaryDirectory(prefix='improve_m2_m4_test_',dir=BASE/'results/paper_v2') as directory:
            experiment=Experiment(directory)
            experiment.step_name='train'; experiment.deadline=time.time()+120
            def graphs(stems,segments):
                return [Data(x=torch.randn(12,15),edge_index=torch.stack([torch.arange(12),torch.roll(torch.arange(12),1)]),
                             pos=torch.rand(12,2),y=torch.arange(12)%2,skeleton_counts=torch.ones(12)) for _ in range(2)]
            experiment.native_graphs=graphs
            save(Path(directory)/'protocol.json',dict(split=dict(train=['synthetic_a'],val=['synthetic_b'])))
            cfg=load_config(BASE/'configs/step2_loss/hybrid_full_focal.yaml')
            cfg['training']['epochs']=3
            run=dict(config=cfg,segments=300,loss='weighted_ce',skeleton_weight=.3)
            result=experiment.train_graph(run,Path(directory))
            self.assertEqual(result['epochs'],3)
            self.assertTrue(Path(result['best_checkpoint']).is_file())
            self.assertGreater(result['projected_full_run_seconds'],0)
            self.assertFalse((Path(directory)/'test_consumed.json').exists())


if __name__=='__main__': unittest.main()
