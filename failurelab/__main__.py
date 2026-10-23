import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .data import DATA_SHA256, TARGET, FEATURES, FORBIDDEN, load_benchmark, split_indices
from .models import make_model, Model
from .policy import metrics, choose_threshold, block_bootstrap, operating_point

ROOT = Path(__file__).resolve().parents[1]

def source_digest():
    files = sorted((ROOT / "failurelab").glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + b"\0" + p.read_text(encoding="utf-8").encode("utf-8") for p in files)).hexdigest()

def train(data, output):
    from .reporting import save_figures, write_summary
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_benchmark(data)
    plan_bytes = (ROOT / "experiment_plan.json").read_text(encoding="utf-8").encode("utf-8")
    plan = json.loads(plan_bytes)
    assert plan["dataset_sha256"] == DATA_SHA256
    parts = split_indices(len(frame))
    for name, (lo, hi) in plan["ranges_udi"].items():
        assert (frame.iloc[parts[name]]["UDI"].iloc[[0, -1]].tolist()) == [lo, hi]
    y = frame[TARGET].to_numpy()
    fitted, selection = {}, {}
    for name in plan["models"]:
        model = make_model(name, plan["seed"])
        model.fit(frame.iloc[parts["train"]], y[parts["train"]])
        score = model.predict_proba(frame.iloc[parts["selection"]])[:, 1]
        selection[name] = metrics(y[parts["selection"]], score)
        fitted[name] = model
        print(f"{name}: selection AP={selection[name]['average_precision']:.4f}", flush=True)
    selected = max(plan["models"], key=lambda name: selection[name]["average_precision"])
    model = fitted[selected]
    policy_scores = model.predict_proba(frame.iloc[parts["policy"]])[:, 1]
    # Fix all decisions before accessing test labels or generating test measurements.
    policies = {cost: choose_threshold(y[parts["policy"]], policy_scores, cost)
                for cost in plan["sensitivity_miss_costs"]}
    threshold = policies[25]["threshold"]
    decision = {"selected_model": selected, "threshold": threshold,
                "selection": selection, "policy_points": policies,
                "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "source_sha256": source_digest()}
    (out / "frozen_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    test = parts["test"]
    test_y = y[test]
    scores = {name: fitted[name].predict_proba(frame.iloc[test])[:, 1] for name in fitted}
    selected_scores = scores[selected]
    result = {
        "dataset_sha256": DATA_SHA256, "experiment_plan_sha256": decision["plan_sha256"],
        "source_sha256": decision["source_sha256"],
        "versions": {key: importlib.metadata.version(key) for key in
                     ["numpy", "pandas"]},
        "python": sys.version.split()[0], "seed": plan["seed"],
        "splits": {name: {"rows": len(ix), "failures": int(y[ix].sum()),
                         "udi_first": int(frame.iloc[ix]["UDI"].iloc[0]),
                         "udi_last": int(frame.iloc[ix]["UDI"].iloc[-1])} for name, ix in parts.items()},
        "input_features": FEATURES, "excluded_columns": FORBIDDEN,
        "selection": selection, "selected_model": selected, "threshold": threshold,
        "policy_block_operating_point": policies[25],
        "test_models": {name: metrics(test_y, score) for name, score in scores.items()},
        "test_default_threshold": metrics(test_y, selected_scores, 0.5),
        "test_policy": metrics(test_y, selected_scores, threshold),
        "baselines": {"never_alert": operating_point(test_y, np.zeros(len(test)), 1.0),
                      "always_alert": operating_point(test_y, np.ones(len(test)), 0.0)},
        "cost_sensitivity": [{"miss_cost": cost, "false_alarm_cost": 1,
                             "threshold": policies[cost]["threshold"],
                             "test": metrics(test_y, selected_scores, policies[cost]["threshold"], cost)}
                            for cost in plan["sensitivity_miss_costs"]],
        "uncertainty": block_bootstrap(test_y, selected_scores, threshold,
                                      **plan["bootstrap"]),
        "limitations": ["Synthetic data; labels represent current failure, not future failure.",
                        "UDI order is not a verified timestamp or machine-group split.",
                        "Raw model scores have not been separately calibrated.",
                        "Cost ratio 25:1 is illustrative; no actual savings or intervention efficacy measured.",
                        "Only 39 positive cases in the fixed test block; uncertainty is substantial."],
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    predictions = pd.DataFrame({"UDI": frame.iloc[test]["UDI"].to_numpy(), "label": test_y,
                                **{f"score_{name}": s for name, s in scores.items()},
                                "alert": (selected_scores >= threshold).astype(int)})
    predictions.to_csv(out / "test_predictions.csv", index=False)
    bundle = {"model": model.to_dict(), "threshold": threshold, "model_name": selected,
              "dataset_sha256": DATA_SHA256}
    (out / "model.json").write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n")
    save_figures(test_y, scores, result, out)
    write_summary(result, out)
    print(json.dumps({"selected": selected, "test": result["test_policy"]}, indent=2))
    return result

def predict(model_path, input_path, output_path):
    bundle = json.loads(Path(model_path).read_text())
    frame = pd.read_csv(input_path)
    score = Model.from_dict(bundle["model"]).predict_proba(frame)[:, 1]
    out = pd.DataFrame({"row": np.arange(len(frame)), "risk_score": score,
                        "alert": (score >= bundle["threshold"]).astype(int),
                        "threshold": bundle["threshold"]})
    if "UDI" in frame:
        out.insert(0, "UDI", frame["UDI"].to_numpy())
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} scored observations to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Machine Failure Cost Lab")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("train", help="Run the fixed four-way benchmark")
    p.add_argument("--data", type=Path, default=ROOT / "data/ai4i2020.csv")
    p.add_argument("--output", type=Path, default=ROOT / "reports")
    p = subs.add_parser("predict", help="Score a sensor CSV with a JSON model artifact")
    p.add_argument("--model", type=Path, default=ROOT / "reports/model.json")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.output)
    else:
        predict(args.model, args.input, args.output)

if __name__ == "__main__":
    main()
