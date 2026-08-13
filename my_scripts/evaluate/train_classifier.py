"""
Improvement #2: combine BOTH signals (surprise_physical + risk_head) into a
real trained classifier (logistic regression), instead of a single
hand-picked threshold on surprise alone.

IMPORTANT: evaluated with 5-fold cross-validation - the classifier is
NEVER tested on data it was trained on. This is the honest way to report
a trained model's performance; testing on training data would give
artificially inflated numbers.

Usage:
    python train_classifier.py --summary test_outputs/multi_surprise_summary.json
"""

import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    per_episode = data["per_episode"]

    X, y, attack_types = [], [], []
    for entry in per_episode.values():
        X.append([entry["surprise_physical"], entry["risk_head"]])
        y.append(1 if entry["condition"] != "normal" else 0)
        attack_types.append(entry["condition"])

    X = np.array(X)
    y = np.array(y)

    print(f"Total episodes: {len(y)}  (attacks: {sum(y)}, normal: {len(y) - sum(y)})")
    print(f"Features: [surprise_physical, risk_head]")
    print(f"Evaluated with {args.folds}-fold stratified cross-validation (honest, no data leakage)\n")

    clf = LogisticRegression(class_weight="balanced")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    # Out-of-fold predictions - each episode is predicted by a model that
    # never saw it during training
    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print("=" * 70)
    print("CROSS-VALIDATED RESULTS (2-feature logistic regression)")
    print("=" * 70)
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  True Positives:  {tp}   False Negatives: {fn}")
    print(f"  False Positives: {fp}   True Negatives:  {tn}")

    print("\n" + "=" * 70)
    print("PER-ATTACK-TYPE RECALL (cross-validated)")
    print("=" * 70)
    per_attack = {}
    for pred, label, attack in zip(y_pred, y, attack_types):
        if attack == "normal":
            continue
        if attack not in per_attack:
            per_attack[attack] = {"caught": 0, "missed": 0}
        if pred == 1:
            per_attack[attack]["caught"] += 1
        else:
            per_attack[attack]["missed"] += 1

    for attack, counts in per_attack.items():
        total = counts["caught"] + counts["missed"]
        pct = 100 * counts["caught"] / total if total else 0
        print(f"  {attack:28s} caught={counts['caught']}/{total} ({pct:.0f}%)")

    # Fit final model on ALL data for reporting learned coefficients
    # (not for evaluation - that's what the CV above is for)
    clf.fit(X, y)
    print("\n" + "=" * 70)
    print("Learned feature weights (fit on all data, for reference only)")
    print("=" * 70)
    print(f"  surprise_physical weight: {clf.coef_[0][0]:+.3f}")
    print(f"  risk_head weight:         {clf.coef_[0][1]:+.3f}")

    result = {
        "auc": auc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
        "per_attack_recall": per_attack,
        "feature_weights": {"surprise_physical": float(clf.coef_[0][0]), "risk_head": float(clf.coef_[0][1])},
    }
    with open("test_outputs/classifier_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to test_outputs/classifier_results.json")


if __name__ == "__main__":
    main()