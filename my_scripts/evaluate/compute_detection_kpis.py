import argparse
import json
import numpy as np


def roc_curve_manual(scores, labels):
    thresholds = sorted(set(scores), reverse=True)
    tpr_list, fpr_list = [], []
    P = sum(labels)
    N = len(labels) - P
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        tpr_list.append(tp / P if P else 0)
        fpr_list.append(fp / N if N else 0)
    return fpr_list, tpr_list, thresholds


def auc_trapezoid(fpr, tpr):
    points = sorted(zip(fpr, tpr))
    area = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Manually chosen threshold. If omitted, automatically picks the best-balance point (Youden's J).")
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    per_episode = data["per_episode"]

    normal_scores = [e["surprise_physical"] for e in per_episode.values() if e["condition"] == "normal"]
    normal_mean = np.mean(normal_scores)
    normal_std = np.std(normal_scores)
    print(f"Normal baseline: mean={normal_mean:.4f}, std={normal_std:.4f}")

    scores, labels, attack_types = [], [], []
    for episode_name, entry in per_episode.items():
        z = abs(entry["surprise_physical"] - normal_mean) / normal_std if normal_std > 0 else 0
        scores.append(z)
        is_attack = 1 if entry["condition"] != "normal" else 0
        labels.append(is_attack)
        attack_types.append(entry["condition"])

    fpr, tpr, thresholds = roc_curve_manual(scores, labels)
    auc = auc_trapezoid(fpr, tpr)

    print("\n" + "=" * 70)
    print(f"ROC-AUC (two-sided deviation score): {auc:.3f}")
    print("=" * 70)

    if args.threshold is not None:
        best_thresh = args.threshold
        tp_manual = sum(1 for s, l in zip(scores, labels) if s >= best_thresh and l == 1)
        fp_manual = sum(1 for s, l in zip(scores, labels) if s >= best_thresh and l == 0)
        P = sum(labels)
        N = len(labels) - P
        tpr_val = tp_manual / P if P else 0
        fpr_val = fp_manual / N if N else 0
        print(f"\nUsing MANUALLY CHOSEN threshold: |z-score| >= {best_thresh:.3f}")
        print(f"  TPR (recall) at this threshold: {tpr_val:.3f}")
        print(f"  FPR at this threshold:          {fpr_val:.3f}")
        best_idx = None
    else:
        best_j, best_thresh, best_idx = -1, None, None
        for i, t in enumerate(thresholds):
            j = tpr[i] - fpr[i]
            if j > best_j:
                best_j, best_thresh, best_idx = j, t, i
        print(f"\nAutomatic best threshold (Youden's J): |z-score| >= {best_thresh:.3f}")
        print(f"  TPR (recall) at this threshold: {tpr[best_idx]:.3f}")
        print(f"  FPR at this threshold:          {fpr[best_idx]:.3f}")

    tp = sum(1 for s, l in zip(scores, labels) if s >= best_thresh and l == 1)
    fn = sum(1 for s, l in zip(scores, labels) if s < best_thresh and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= best_thresh and l == 0)
    tn = sum(1 for s, l in zip(scores, labels) if s < best_thresh and l == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\nConfusion matrix at best threshold:")
    print(f"  True Positives:  {tp}   False Negatives: {fn}")
    print(f"  False Positives: {fp}   True Negatives:  {tn}")
    print(f"\nPrecision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    print("\n" + "=" * 70)
    print("PER-ATTACK-TYPE RECALL at this threshold")
    print("=" * 70)
    per_attack = {}
    for s, l, a in zip(scores, labels, attack_types):
        if a == "normal":
            continue
        if a not in per_attack:
            per_attack[a] = {"caught": 0, "missed": 0}
        if s >= best_thresh:
            per_attack[a]["caught"] += 1
        else:
            per_attack[a]["missed"] += 1

    for attack, counts in per_attack.items():
        total = counts["caught"] + counts["missed"]
        recall_pct = 100 * counts["caught"] / total if total else 0
        print(f"  {attack:28s} caught={counts['caught']}/{total} ({recall_pct:.0f}%)")

    result = {
        "auc": auc, "threshold": best_thresh,
        "threshold_mode": "manual" if args.threshold is not None else "automatic (Youden's J)",
        "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "per_attack_recall": per_attack,
    }
    with open("test_outputs/detection_kpis_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to test_outputs/detection_kpis_v2.json")


if __name__ == "__main__":
    main()