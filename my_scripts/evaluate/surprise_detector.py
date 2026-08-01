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
    None: "Driving normally, no hazard detected. Maintaining steady speed.",
}


def build_state_vector(features, encoder):
    mapped_scenario = features.get("mapped_scenario")
    steering, throttle, brake = 0.0, 0.3, 0.0  # approximated, see earlier note
    driving_state = np.array([
        steering, throttle, brake,
        float(features.get("speed_ratio", 0.0)),
        float(features.get("red_light", 0)),
        float(features.get("near_stop", 0)),
    ], dtype=np.float32)
    cot_text = COT_TEXT_BY_SCENARIO.get(mapped_scenario)
    cot_vec = encoder.encode(cot_text, convert_to_tensor=False).astype(np.float32)
    return np.concatenate([driving_state, cot_vec]), cot_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surprise-data", required=True)
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

    with open(args.surprise_data) as f:
        data = json.load(f)

    results = {}
    print("\n" + "=" * 70)
    print("SURPRISE SCORES: real prediction error, normal vs attack")
    print("=" * 70)

    for episode_name, entry in data.items():
        state_t_vec, cot_text_t = build_state_vector(entry["t"], encoder)
        state_t1_vec, cot_text_t1 = build_state_vector(entry["t1"], encoder)

        state_t = torch.tensor(state_t_vec, dtype=torch.float32, device=device).unsqueeze(0)
        actual_t1 = torch.tensor(state_t1_vec, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            raw_mean, _, _ = policy.forward(state_t)
            action = squash_to_action(torch.tanh(raw_mean), device)
            predicted_t1, risk_hat, _ = world_model(state_t, action)

            surprise_full = torch.nn.functional.mse_loss(predicted_t1, actual_t1).item()
           
            surprise_physical = torch.nn.functional.mse_loss(
                predicted_t1[:, :6], actual_t1[:, :6]
            ).item()

        print(f"\nEpisode: {episode_name}")
        print(f"  Attack: {entry.get('attack')}")
        print(f"  CoT text at t:  \"{cot_text_t}\"")
        print(f"  CoT text at t1: \"{cot_text_t1}\"")
        print(f"  Risk head output (from state t): {risk_hat.item():.3f}")
        print(f"  SURPRISE SCORE, full 390-dim (prediction error): {surprise_full:.4f}")
        print(f"  SURPRISE SCORE, physical-only 6-dim (prediction error): {surprise_physical:.4f}")

        results[episode_name] = {
            "attack": entry.get("attack"),
            "frame": entry.get("frame_t"),
            "surprise_score_full": surprise_full,
            "surprise_score_physical": surprise_physical,
            "risk_head": risk_hat.item(),
        }

    with open("test_outputs/surprise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to test_outputs/surprise_results.json")


if __name__ == "__main__":
    main()