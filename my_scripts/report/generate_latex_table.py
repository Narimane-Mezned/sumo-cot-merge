import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    s = data["summary"]
    sig = data.get("significance", {})

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Condition} & \textbf{n} & \textbf{Surprise (physical)} & \textbf{Risk head} & \textbf{p-value (surprise / risk)} \\")
    print(r"\midrule")

    # normal first, then everything else
    ordered = ["normal"] + [k for k in s.keys() if k != "normal"]
    for condition in ordered:
        row = s[condition]
        n = row["n_episodes"]
        surprise = f"{row['surprise_physical_mean']:.4f} $\\pm$ {row['surprise_physical_std']:.4f}"
        risk = f"{row['risk_head_mean']:.3f} $\\pm$ {row['risk_head_std']:.3f}"
        if condition == "normal":
            p_str = "--"
        else:
            p_surprise = sig.get(condition, {}).get("surprise_physical", {}).get("p")
            p_risk = sig.get(condition, {}).get("risk_head", {}).get("p")
            p_surprise_str = f"{p_surprise:.3f}" if p_surprise is not None else "n/a"
            p_risk_str = f"{p_risk:.3f}" if p_risk is not None else "n/a"
            p_str = f"{p_surprise_str} / {p_risk_str}"
        label = condition.replace("_", "\\_")
        print(f"{label} & {n} & {surprise} & {risk} & {p_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Surprise and risk statistics, all attack types vs.\ normal driving. "
          r"p-values from Welch's t-test against the normal condition.}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()