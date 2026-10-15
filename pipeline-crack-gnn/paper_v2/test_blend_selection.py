"""Blend selector regressions using synthetic arrays only; no dataset access."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import improve_m2_m4 as driver


class BlendSelectionTests(unittest.TestCase):
    def select(self, score, thresholds):
        class Metrics:
            def score(self, predictions):
                alpha, low, high = list(predictions)[0]
                return dict(strict_cr=score(alpha, low, high), cldice=.7, pixel_f1=.7)

        with tempfile.TemporaryDirectory(prefix='improve_m2_m4_test_',
                                         dir=driver.BASE/'results/paper_v2') as directory:
            experiment = driver.Experiment(directory)
            experiment.step_name = 'train'
            experiment.deadline = time.time()+120
            baseline = dict(recipe=dict(low=.5, high=.5, min_area=0, max_gap=0),
                            metrics=dict(strict_cr=.3, cldice=.7, pixel_f1=.7))
            def encode(probabilities, recipe):
                return float(probabilities.mean()), recipe.low, recipe.high
            with patch.object(driver, 'process', encode), patch.dict(driver.GRIDS,
                    dict(thresholds=thresholds, alphas=[0., .1, 1.], min_areas=[0], max_gaps=[0])):
                result = experiment.search_blend([np.ones((2,2), np.float32)],
                    [np.zeros((2,2), np.float32)], Metrics(), baseline, 'synthetic', .7)
            self.assertFalse((Path(directory)/'test_consumed.json').exists())
            return result

    def test_superior_zero_weight_control_is_retained(self):
        result = self.select(lambda alpha, low, high:
                             .4 if alpha == 0 else (.35 if alpha < .2 else .3), [.5])
        self.assertEqual(result['alpha'], 0.)
        self.assertEqual(result['metrics']['strict_cr'], .4)

    def test_equal_metrics_prefer_fewer_processing_operations(self):
        def score(alpha, low, high):
            return .4 if alpha == 0 and (low, high) in ((.4,.5),(.6,.6)) else .3
        result = self.select(score, [.4,.5,.6])
        self.assertEqual(result['alpha'], 0.)
        self.assertEqual((result['recipe']['low'], result['recipe']['high']), (.6,.6))


if __name__ == '__main__':
    unittest.main()
