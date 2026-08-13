import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    per_episode = data["per_episode"]
    normal_scores = [e["surprise_physical"] for e in per_episode.values() if e["condition"] == "normal"]
    normal_mean = np.mean(normal_scores)
    normal_std = np.std(normal_scores)

    scores, labels, attack_types = [], [], []
    for entry in per_episode.values():
        z = abs(entry["surprise_physical"] - normal_mean) / normal_std if normal_std > 0 else 0
        scores.append(z)
        labels.append(1 if entry["condition"] != "normal" else 0)
        attack_types.append(entry["condition"])

    thresholds = sorted(set(scores))
    P = sum(labels)
    N = len(labels) - P

    print("=" * 90)
    print(f"{'threshold':>10} {'recall':>8} {'FPR':>8} {'precision':>10} {'missed attacks (by type)':>40}")
    print("=" * 90)

    rows = []
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fn = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        recall = tp / P if P else 0
        fpr = fp / N if N else 0
        precision = tp / (tp + fp) if (tp + fp) else 1.0

        missed = {}
        for s, l, a in zip(scores, labels, attack_types):
            if l == 1 and s < t:
                missed[a] = missed.get(a, 0) + 1
        missed_str = ", ".join(f"{k}:{v}" for k, v in missed.items()) if missed else "none"

        rows.append((t, recall, fpr, precision, missed_str))
        print(f"{t:>10.3f} {recall:>8.1%} {fpr:>8.1%} {precision:>10.1%} {missed_str:>40}")

    
    max_recall_row = max(rows, key=lambda r: r[1])
    print("\n" + "=" * 90)
    print("MAXIMUM RECALL OPERATING POINT (catches the most attacks possible)")
    print("=" * 90)
    t, recall, fpr, precision, missed_str = max_recall_row
    print(f"Threshold: |z| >= {t:.3f}")
    print(f"Recall: {recall:.1%}   FPR: {fpr:.1%}   Precision: {precision:.1%}")
    print(f"Missed attacks at this point: {missed_str}")

    if recall < 1.0:
        print("\nNote: recall < 100% even at the lowest usable threshold because some")
        print("attack episodes' surprise scores fall inside the normal driving range")
        print("itself - no threshold on this single signal can separate them further.")
        print("This is a real limit of a single-signal detector, and the honest next")
        print("step is combining this with a second, independent signal (e.g. the VLM")
        print("reasoning output), rather than claiming a threshold fixes it.")


if __name__ == "__main__":
    main()