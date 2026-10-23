"""Exact SVG charts from recorded values; no plotting binary is required."""
from pathlib import Path
from html import escape
import numpy as np
from .ranking import curve

def svg_text(x, y, text, size=15, color="#34445c", anchor="start"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-family="Arial,sans-serif">{escape(str(text))}</text>'

def svg_start(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
            f"<title>{escape(title)}</title><desc>{escape(description)}</desc>",
            f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
            svg_text(36, 42, title, 24, "#11243c"),
            svg_text(36, 67, description, 13)]

def save_figures(y, scores, result, out):
    out = Path(out)
    colors = ["#7b8797", "#6558d3", "#c67e13", "#008579"]
    parts = svg_start(900, 570, "Precision and recall on the held-out block",
                      "2,000 observations | 39 failures | UDI 8001-10000 | model selected on a separate block")
    left, top, width, height = 75, 108, 565, 365
    for value in np.linspace(0, 1, 6):
        x, yy = left+value*width, top+(1-value)*height
        parts += [f'<path d="M{left},{yy} H{left+width} M{x},{top} V{top+height}" stroke="#e0e7ef" fill="none"/>',
                  svg_text(left-12, yy+5, f"{value:.1f}", 12, anchor="end"),
                  svg_text(x, top+height+23, f"{value:.1f}", 12, anchor="middle")]
    for i, ((name, score), color) in enumerate(zip(scores.items(), colors)):
        recall, precision, _ = curve(y, score)
        coords = " ".join(f"{left+r*width:.2f},{top+(1-p)*height:.2f}" for r,p in zip(recall, precision))
        parts += [f'<polyline points="{coords}" stroke="{color}" stroke-width="2" fill="none"/>',
                  svg_text(665, 128+i*58, name, 14, color),
                  svg_text(665, 149+i*58, f"AP {result['test_models'][name]['average_precision']:.4f}", 13)]
    yy = top+(1-float(np.mean(y)))*height
    parts += [f'<path d="M{left},{yy} H{left+width}" stroke="#778392" stroke-dasharray="4 4" fill="none"/>',
              svg_text(357, 526, "Recall", 15, anchor="middle"),
              svg_text(75, 96, "Precision", 14),
              svg_text(665, 388, "Dashed: prevalence", 12),
              svg_text(36, 554, "Synthetic current-failure classification. AP uses threshold groups; the connecting line is a visual guide.", 12)]
    (out/"precision_recall.svg").write_text("\n".join(parts+["</svg>"]), encoding="utf-8")
    p = result["test_policy"]
    parts = svg_start(900, 490, "What the chosen alert policy does",
                      "Illustrative cost = 25 x missed failures + false alerts; this is not measured money saved.")
    points = [result["baselines"]["never_alert"], result["baselines"]["always_alert"],
              result["test_default_threshold"], p]
    labels = ["Never alert", "Always alert", "Threshold 0.5", "Chosen policy"]
    ymax=max(point["cost"] for point in points)*1.15
    for i,(label,point,color) in enumerate(zip(labels,points,["#9ca9b9","#9ca9b9","#6558d3","#008579"])):
        x=62+i*132
        height=250*point["cost"]/ymax
        parts += [f'<rect x="{x}" y="{370-height:.2f}" width="85" height="{height:.2f}" fill="{color}" rx="4"/>',
                  svg_text(x+42, 358-height, f"{point['cost']:.0f}", 17, anchor="middle"),
                  svg_text(x+42, 394, label, 13, anchor="middle")]
    parts += [svg_text(60,105,"Total assumed cost units",15)]
    for i,(label,key,color) in enumerate([("Detected failures","tp","#008579"),
                                         ("Missed failures","fn","#be4b4b"),
                                         ("False alerts","fp","#a56b15")]):
        parts += [svg_text(665,145+i*88,p[key],32,color),
                  svg_text(665,169+i*88,label,14)]
    parts += [svg_text(36,443,f"Threshold {result['threshold']:.6f} | recall {p['recall']:.1%} | precision {p['precision']:.1%} | alert rate {p['alert_rate']:.1%}",15),
              svg_text(36,470,"Threshold chosen on UDI 7001-8000. These measurements use the separate UDI 8001-10000 test block.",12)]
    (out/"alert_policy.svg").write_text("\n".join(parts+["</svg>"]), encoding="utf-8")

def write_summary(result, out):
    rows = ["# Recorded benchmark results", "",
            "Synthetic contemporaneous failure classification; no advance-warning claim.", "",
            "| Model | Selection AP | Test AP | Test ROC-AUC | Test Brier |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in result["test_models"].items():
        rows.append(f"| {name} | {result['selection'][name]['average_precision']:.4f} | "
                    f"{m['average_precision']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |")
    p = result["test_policy"]
    rows += ["", f"Model selected before test evaluation: **{result['selected_model']}**.",
             f"Policy threshold selected on UDI 7001-8000: **{result['threshold']:.6f}**.",
             "", f"Test: {p['tp']} detected failures, {p['fn']} missed failures, {p['fp']} false alerts; "
             f"recall {p['recall']:.1%}, precision {p['precision']:.1%}, alert rate {p['alert_rate']:.1%}.",
             "", f"Illustrative cost: **{p['cost']:.0f} units**, compared with "
             f"{result['baselines']['never_alert']['cost']:.0f} for never alerting and "
             f"{result['test_default_threshold']['cost']:.0f} at the model's default 0.5 threshold.",
             "", "Costs are assumptions, not measured maintenance expenses or money saved.",
             "", "## Cost sensitivity", "",
             "| Miss cost / false-alert cost | Policy-set threshold | Test cost | Recall | False alerts |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for row in result["cost_sensitivity"]:
        m = row["test"]
        rows.append(f"| {row['miss_cost']}:1 | {row['threshold']:.6f} | {m['cost']:.0f} | "
                    f"{m['recall']:.1%} | {m['fp']} |")
    rows += ["", "Every sensitivity threshold was selected on the policy block, not the test block.",
             "", "## Conditional 95% uncertainty intervals", ""]
    for name, bounds in result["uncertainty"]["intervals_95"].items():
        rows.append(f"- {name}: {bounds[0]:.4f} to {bounds[1]:.4f}")
    rows += ["", "These ordered-block bootstrap intervals condition on this fitted model and policy. "
             "They do not include model-selection, training, or deployment-shift uncertainty.",
             "", "Full measurements and provenance: [results.json](results.json).",
             ""]
    (Path(out) / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
