import argparse
import json
import numpy as np
import torch
import torch.nn as nn

ACTION_LOW = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        z = self.shared(s)
        raw_mean = self.actor_mean(z)
        log_std = self.actor_log_std.clamp(-3.0, 1.0)
        std = log_std.exp().expand_as(raw_mean)
        value = self.critic(z).squeeze(-1)
        return raw_mean, std, value


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.progress_head = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.net(x)
        s_next_hat = self.next_state(z)
        risk_hat = self.risk_head(z).squeeze(-1)
        progress_hat = self.progress_head(z).squeeze(-1)
        return s_next_hat, risk_hat, progress_hat


def squash_to_action(raw_tanh, device):
    low = torch.tensor(ACTION_LOW, device=device)
    high = torch.tensor(ACTION_HIGH, device=device)
    return low + (raw_tanh + 1.0) * 0.5 * (high - low)


COT_TEXT_BY_SCENARIO = {
    "red_light": "The traffic light ahead is RED. I must stop before the line. Applying brake now.",
    "near_stop": "A stop sign is nearby. Slowing down and preparing to stop completely.",
    "hard_brake": "Sudden obstacle or hazard detected ahead. Emergency braking applied immediately.",
    "traffic_jam": "Traffic congestion detected. Switching to stop-and-go mode at low speed.",
    None: "Driving normally, no hazard detected. Maintaining steady speed.",
}


def build_state_vector(features, encoder):
    mapped_scenario = features.get("mapped_scenario")
    steering, throttle, brake = 0.0, 0.3, 0.0
    driving_state = np.array([
        steering, throttle, brake,
        float(features.get("speed_ratio", 0.0)),
        float(features.get("red_light", 0)),
        float(features.get("near_stop", 0)),
    ], dtype=np.float32)
    cot_text = COT_TEXT_BY_SCENARIO.get(mapped_scenario)
    cot_vec = encoder.encode(cot_text, convert_to_tensor=False).astype(np.float32)
    return np.concatenate([driving_state, cot_vec])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encoder.eval()

    state_dim, action_dim = 6 + 384, 3
    policy = ActorCritic(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(f"{args.checkpoint_dir}/dreamer_ppo_policy_best.pt", map_location=device, weights_only=True))
    policy.eval()

    world_model = WorldModel(state_dim, action_dim).to(device)
    world_model.load_state_dict(torch.load(f"{args.checkpoint_dir}/dreamer_ppo_worldmodel_best.pt", map_location=device, weights_only=True))
    world_model.eval()

    with open(args.data) as f:
        data = json.load(f)

    per_episode = {}
    grouped = {}

    for episode_name, entry in data.items():
        condition = entry.get("attack", "normal")
        if condition not in grouped:
            grouped[condition] = {"surprise_physical": [], "surprise_full": [], "risk_head": []}

        state_t_vec = build_state_vector(entry["t"], encoder)
        state_t1_vec = build_state_vector(entry["t1"], encoder)
        state_t = torch.tensor(state_t_vec, dtype=torch.float32, device=device).unsqueeze(0)
        actual_t1 = torch.tensor(state_t1_vec, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            raw_mean, _, _ = policy.forward(state_t)
            action = squash_to_action(torch.tanh(raw_mean), device)
            predicted_t1, risk_hat, _ = world_model(state_t, action)
            surprise_full = torch.nn.functional.mse_loss(predicted_t1, actual_t1).item()
            surprise_physical = torch.nn.functional.mse_loss(predicted_t1[:, :6], actual_t1[:, :6]).item()

        per_episode[episode_name] = {
            "condition": condition,
            "surprise_full": surprise_full,
            "surprise_physical": surprise_physical,
            "risk_head": risk_hat.item(),
        }
        grouped[condition]["surprise_physical"].append(surprise_physical)
        grouped[condition]["surprise_full"].append(surprise_full)
        grouped[condition]["risk_head"].append(risk_hat.item())

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS PER ATTACK TYPE (mean ± std)")
    print("=" * 70)

    summary = {}
    for condition, vals in grouped.items():
        n = len(vals["surprise_physical"])
        summary[condition] = {
            "n_episodes": n,
            "surprise_physical_mean": float(np.mean(vals["surprise_physical"])) if n else None,
            "surprise_physical_std": float(np.std(vals["surprise_physical"])) if n else None,
            "surprise_full_mean": float(np.mean(vals["surprise_full"])) if n else None,
            "surprise_full_std": float(np.std(vals["surprise_full"])) if n else None,
            "risk_head_mean": float(np.mean(vals["risk_head"])) if n else None,
            "risk_head_std": float(np.std(vals["risk_head"])) if n else None,
        }
        print(f"\n{condition.upper()} (n={n})")
        print(f"  Surprise (physical-only): {summary[condition]['surprise_physical_mean']:.4f} ± {summary[condition]['surprise_physical_std']:.4f}")
        print(f"  Risk head:                {summary[condition]['risk_head_mean']:.3f} ± {summary[condition]['risk_head_std']:.3f}")

    from scipy import stats as sstats

    print("\n" + "=" * 70)
    print("SIGNIFICANCE CHECK: each attack vs normal (Welch's t-test)")
    print("=" * 70)
    normal_vals = grouped.get("normal", {"surprise_physical": [], "risk_head": []})
    significance = {}
    for condition, vals in grouped.items():
        if condition == "normal":
            continue
        row = {}
        for metric in ["surprise_physical", "risk_head"]:
            a, b = vals[metric], normal_vals[metric]
            if len(a) >= 2 and len(b) >= 2:
                t_stat, p_value = sstats.ttest_ind(a, b, equal_var=False)
                row[metric] = {"t": float(t_stat), "p": float(p_value)}
                verdict = "real difference" if p_value < 0.05 else "not confirmed"
                print(f"  {condition:28s} {metric:20s} t={t_stat:+.2f} p={p_value:.4f} -> {verdict}")
        significance[condition] = row

    with open("test_outputs/multi_surprise_summary.json", "w") as f:
        json.dump({"per_episode": per_episode, "summary": summary, "significance": significance}, f, indent=2)
    print("\nSaved to test_outputs/multi_surprise_summary.json")


if __name__ == "__main__":
    main()