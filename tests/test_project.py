import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from failurelab.data import FEATURES, NUMERIC, ENGINEERED, TARGET, SensorFeatures, validate_features, split_indices, load_benchmark
from failurelab.models import make_model, Model, fit_tree, tree_predict
from failurelab.policy import choose_threshold, operating_point, block_bootstrap, metrics

ROOT = Path(__file__).resolve().parents[1]
def sample(n=60):
    rng = np.random.default_rng(8)
    return pd.DataFrame({"Type": np.resize(["L", "M", "H"], n),
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 1, n),
        "Rotational speed [rpm]": rng.uniform(1000, 2000, n),
        "Torque [Nm]": rng.uniform(15, 75, n),
        "Tool wear [min]": rng.uniform(0, 240, n)})

class FeatureTests(unittest.TestCase):
    def test_target_and_identifier_columns_cannot_change_features(self):
        x = sample()
        polluted = x.assign(UDI=-999, **{"Product ID": "LEAK", TARGET: 1,
            "TWF": 1, "HDF": 1, "PWF": 1, "OSF": 1, "RNF": 1})
        for engineered in [False, True]:
            f = SensorFeatures(engineered)
            assert_frame_equal(f.fit_transform(x), f.fit_transform(polluted))

    def test_physical_units(self):
        x = sample(1)
        x.loc[0, NUMERIC] = [300, 310, 1200, 40, 200]
        z = SensorFeatures(True).fit_transform(x)
        assert_allclose(z[ENGINEERED].iloc[0].to_numpy(float),
                        [10, 1600 * np.pi, 8000])

    def test_bad_sensor_inputs_are_rejected(self):
        x = sample()
        for bad in [x.drop(columns=[NUMERIC[0]]), x.assign(Type="X"),
                    x.assign(**{NUMERIC[0]: np.nan}),
                    x.assign(**{NUMERIC[2]: 0}),
                    x.assign(**{NUMERIC[3]: -1})]:
            with self.assertRaises(ValueError):
                validate_features(bad)

    def test_scaler_is_fitted_only_on_training_rows(self):
        x = sample()
        y = np.arange(len(x)) % 2
        model = make_model("logistic").fit(x, y)
        assert_allclose(model.mean[:5], x[NUMERIC].mean().to_numpy())
        old_mean = model.mean.copy()
        shifted = x.copy()
        shifted[NUMERIC[0]] += 1000
        model.predict_proba(shifted)
        assert_allclose(model.mean, old_mean)

class ProtocolTests(unittest.TestCase):
    def test_splits_cover_data_once_in_order(self):
        parts = split_indices()
        combined = np.concatenate(list(parts.values()))
        np.testing.assert_array_equal(combined, np.arange(10000))
        sets = [set(ix) for ix in parts.values()]
        for i in range(len(sets)):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        plan = json.loads((ROOT / "experiment_plan.json").read_text())
        for name, ix in parts.items():
            self.assertEqual([int(ix[0] + 1), int(ix[-1] + 1)], plan["ranges_udi"][name])

    def test_actual_dataset_contract(self):
        frame = load_benchmark(ROOT / "data/ai4i2020.csv")
        self.assertEqual(frame.shape, (10000, 14))
        self.assertEqual(int(frame[TARGET].sum()), 339)

    def test_modified_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT.resolve()))
            path = Path(folder) / "changed.csv"
            path.write_text("not the benchmark")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_benchmark(path)

class PolicyTests(unittest.TestCase):
    def test_threshold_minimizes_known_cost(self):
        y = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.2]
        best = choose_threshold(y, scores, miss_cost=5)
        self.assertEqual(best["cost"], 1)
        self.assertEqual(best["threshold"], 0.7)
        self.assertEqual((best["tp"], best["fp"], best["fn"], best["tn"]), (2, 1, 0, 1))

    def test_tied_scores_cannot_be_split(self):
        p = choose_threshold([1, 0], [0.5, 0.5], miss_cost=1)
        self.assertEqual(p["cost"], 1)
        self.assertEqual(p["alert_rate"], 0)
        self.assertGreater(p["threshold"], 1)

    def test_threshold_equality_is_an_alert(self):
        p = operating_point([1, 0, 1, 0], [0.5, 0.6, 0.1, 0.2], 0.5)
        self.assertEqual((p["tp"], p["fp"], p["fn"], p["tn"]), (1, 1, 1, 1))
        self.assertEqual(p["cost"], 26)

    def test_invalid_policy_inputs_fail(self):
        for y, p in [([0], [np.nan]), ([0], [2]), ([2], [0.5]), ([0, 1], [0.5]), ([], [])]:
            with self.assertRaises(ValueError):
                choose_threshold(y, p)
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [0.1, 0.9], miss_cost=-1)

    def test_block_bootstrap_is_reproducible(self):
        y = np.tile([0, 1, 0, 0, 0], 20)
        p = np.linspace(0, 1, len(y))
        a = block_bootstrap(y, p, 0.5, repeats=40, block_size=10)
        b = block_bootstrap(y, p, 0.5, repeats=40, block_size=10)
        self.assertEqual(a, b)
        self.assertEqual(a["valid_repeats"], 40)

class IntegrationTests(unittest.TestCase):
    def test_trusted_model_roundtrip_and_label_independence(self):
        x = sample()
        y = (x[NUMERIC[3]] > 50).astype(int)
        model = make_model("logistic").fit(x, y)
        original = model.predict_proba(x)
        polluted = x.assign(**{TARGET: 1-y, "TWF": 1-y})
        assert_allclose(model.predict_proba(polluted), original)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT.resolve()))
            path = Path(folder) / "model.json"
            path.write_text(json.dumps(model.to_dict()))
            restored = Model.from_dict(json.loads(path.read_text()))
            assert_allclose(restored.predict_proba(x), original)

class ReferenceMathTests(unittest.TestCase):
    def test_ranking_metrics_against_hand_calculation(self):
        m = metrics([1,0,1,0], [0.9,0.8,0.7,0.1])
        self.assertAlmostEqual(m["average_precision"], (1+2/3)/2)
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        self.assertAlmostEqual(m["brier_score"], (0.01+0.64+0.09+0.01)/4)

    def test_tied_and_reversed_rankings(self):
        tied = metrics([1,0,1,0], [0.5]*4)
        self.assertAlmostEqual(tied["average_precision"], 0.5)
        self.assertAlmostEqual(tied["roc_auc"], 0.5)
        reversed_ = metrics([1,1,0,0], [0.1,0.2,0.8,0.9])
        self.assertEqual(reversed_["roc_auc"], 0)
        self.assertAlmostEqual(reversed_["average_precision"], (1/3+2/4)/2)

    def test_auc_matches_pairwise_definition_with_ties(self):
        rng = np.random.default_rng(100)
        y = np.tile([0,1,0], 10)
        s = rng.integers(0,5,len(y))/5
        comparison = s[y==1,None]-s[None,y==0]
        expected = np.mean((comparison>0)+0.5*(comparison==0))
        self.assertAlmostEqual(metrics(y,s)["roc_auc"], expected)

    def test_newton_tree_leaves_have_expected_values(self):
        bins = np.array([[0],[0],[1],[1]])
        g = np.array([0.5,0.5,-0.5,-0.5])
        h = np.full(4,0.25)
        tree = fit_tree(bins,g,h,depth=1,min_leaf=1,l2=1)
        assert_allclose(tree_predict(tree,bins), [-2/3,-2/3,2/3,2/3])
        self.assertEqual(tree["bin"],0)

    def test_tree_respects_minimum_leaf(self):
        bins = np.array([[0],[1],[1],[1]])
        tree = fit_tree(bins,np.array([1,-1,-1,-1]),np.ones(4),min_leaf=2)
        self.assertIn("value",tree)

    def test_models_learn_simple_sensor_signal_and_json_roundtrip(self):
        x = sample(120)
        y = (x[NUMERIC[3]] > 50).astype(int)
        for name in ["logistic","boosting_raw","boosting_physics"]:
            model = make_model(name).fit(x,y)
            score = model.predict_proba(x)[:,1]
            self.assertGreater(metrics(y,score)["roc_auc"],0.98)
            restored = Model.from_dict(json.loads(json.dumps(model.to_dict())))
            assert_allclose(restored.predict_proba(x)[:,1],score,atol=0,rtol=0)
            polluted = x.assign(**{TARGET: 1-y,"HDF":1})
            assert_allclose(restored.predict_proba(polluted)[:,1],score)

    def test_prevalence_baseline_is_training_prior(self):
        x = sample(60)
        y = np.r_[np.ones(10),np.zeros(50)]
        m = make_model("prevalence").fit(x,y)
        assert_allclose(m.predict_proba(x)[:,1],1/6)

if __name__ == "__main__":
    unittest.main()
