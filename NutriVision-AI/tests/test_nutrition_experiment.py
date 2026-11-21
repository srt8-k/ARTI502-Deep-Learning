import unittest

import torch

from tools.compare_nutrition import make_head, predict, train, validation_cohort, validation_groups


class ExperimentTests(unittest.TestCase):
    def test_validation_keeps_midnight_capture_sequences_together(self):
        stamps = [86390, 86401, 86450, 172900, 173500, 260000]
        rows = [(f"dish_{stamp:010d}", None, None, None, 1) for stamp in stamps]
        groups = validation_groups(rows)
        self.assertEqual([len(g) for g in groups], [3, 2, 1])
        chosen = validation_cohort(rows, None)
        self.assertTrue(chosen)
        self.assertEqual(chosen, validation_cohort(rows[::-1], None))
        self.assertTrue(all(not (g & chosen and g - chosen) for g in groups))

    def test_candidate_exports_reproduce_two_outputs_on_unseen_features(self):
        # Synthetic tensors exercise the experiment code, not nutrition accuracy.
        rng = torch.Generator().manual_seed(7)
        features = torch.randn(15, 8, generator=rng)
        probs = torch.randn(15, 3, generator=rng).softmax(1)
        labels = 1 + torch.rand(15, 2, generator=rng) * 100
        fit = torch.arange(15) < 10
        for kind in ("direct", "food-aware", "density"):
            with self.subTest(kind=kind):
                head, bundle = train(kind, features, probs, labels, fit, ~fit)
                restored = make_head(bundle["x_mean"].numel()).eval()
                restored.load_state_dict(bundle["weights"])
                with torch.no_grad():
                    predictions = predict(restored, bundle, features[~fit], probs[~fit])
                    torch.testing.assert_close(predictions, predict(head, bundle, features[~fit], probs[~fit]))
                self.assertEqual(tuple(predictions.shape), (5, 2))
                self.assertTrue(torch.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())
