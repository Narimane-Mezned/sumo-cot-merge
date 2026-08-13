"""
LIVE DEMO for filming: runs SUMO-GUI visibly, prints the detector's live
status in the terminal, and triggers a real attack partway through the
recording. Record both the SUMO window and this terminal together (e.g.
side by side) with your screen recorder.

Uses the real trained checkpoint + the real threshold from your 122-episode
result. Choose an attack with 100% recall for a reliable demo:
sensor_spoofing or fake_safety are recommended.

Usage:
    python live_demo.py --attack sensor_spoofing --checkpoint-dir outputs
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import traci

# --- Robust paths, same pattern as the capture scripts ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SUMOCFG = os.path.join(PROJECT_ROOT, "maps", "basic_simulation", "osm.sumocfg")

VIEW_ID = "View #0"
BOX_SIZE = 60
CHECK_INTERVAL = 8        # simulation steps between each detector check
NORMAL_PHASE_CHECKS = 5   # how many calm checks before triggering the attack
PAUSE_BETWEEN_CHECKS = 1.5  # seconds - paces the demo for recording

# Baked-in from your real 122-episode result (test_outputs/multi_surprise_summary.json)
NORMAL_MEAN = 0.2097
NORMAL_STD = 0.0534
THRESHOLD_Z = 0.902  # your confirmed operating threshold

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
        return self.next_state(z), self.risk_head(z).squeeze(-1), self.progress_head(z).squeeze(-1)


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

ATTACK_SCENARIOS = {
    "traffic_light_tampering": "red_light",
    "sensor_spoofing": "hard_brake",
    "universal_perturbation": "hard_brake",
    "fake_safety": "near_stop",
    "fake_emergency": "near_stop",
    "sybil": "traffic_jam",
}


def get_state_features(veh_id):
    if veh_id not in traci.vehicle.getIDList():
        return None
    speed = traci.vehicle.getSpeed(veh_id)
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        max_speed = traci.lane.getMaxSpeed(lane_id)
        speed_ratio = min(1.0, speed / max_speed) if max_speed > 0 else 0.0
    except Exception:
        speed_ratio = 0.0
    red_light = 0
    try:
        next_tls = traci.vehicle.getNextTLS(veh_id)
        if next_tls:
            _, _, distance, state = next_tls[0]
            if distance < 50 and state.lower() == "r":
                red_light = 1
    except Exception:
        pass
    return {"speed_ratio": round(speed_ratio, 3), "red_light": red_light}


def get_valid_vehicle():
    ids = [v for v in traci.vehicle.getIDList() if not v.startswith(("phantom", "sybil"))]
    return random.choice(ids) if ids else None


def build_state_vector(features, mapped_scenario, encoder, near_stop=0):
    steering, throttle, brake = 0.0, 0.3, 0.0
    driving_state = np.array([
        steering, throttle, brake,
        float(features.get("speed_ratio", 0.0)),
        float(features.get("red_light", 0)),
        float(near_stop),
    ], dtype=np.float32)
    cot_text = COT_TEXT_BY_SCENARIO.get(mapped_scenario)
    cot_vec = encoder.encode(cot_text, convert_to_tensor=False).astype(np.float32)
    return np.concatenate([driving_state, cot_vec])


def trigger_attack(attack_name, veh_id):
    print(f"\n{'='*60}")
    print(f"  >>> TRIGGERING ATTACK: {attack_name} <<<")
    print(f"{'='*60}\n")
    near_stop = 0

    if veh_id not in traci.vehicle.getIDList():
        print(f"  (tracked vehicle {veh_id} left the map right before the attack could be applied - skipping this trigger, will retry next cycle)")
        return near_stop, False

    try:
        if attack_name == "traffic_light_tampering":
            for tls_id in traci.trafficlight.getIDList():
                n = len(traci.trafficlight.getRedYellowGreenState(tls_id))
                traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n)
        elif attack_name == "sensor_spoofing":
            traci.vehicle.setSpeed(veh_id, 0.0)
        elif attack_name == "universal_perturbation":
            traci.vehicle.slowDown(veh_id, 0.0, 20.0)
        elif attack_name == "fake_safety":
            near_stop = 1
            traci.vehicle.slowDown(veh_id, 0.5, 20.0)
        elif attack_name == "fake_emergency":
            near_stop = 1
            traci.vehicle.slowDown(veh_id, 1.0, 20.0)
        elif attack_name == "sybil":
            max_speed = traci.vehicle.getMaxSpeed(veh_id)
            traci.vehicle.slowDown(veh_id, max_speed * 0.15, 20.0)
    except traci.exceptions.TraCIException as e:
        print(f"  (vehicle disappeared mid-command, skipping this trigger: {e})")
        return near_stop, False

    return near_stop, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=list(ATTACK_SCENARIOS.keys()))
    parser.add_argument("--checkpoint-dir", default="outputs")
    args = parser.parse_args()

    mapped_scenario = ATTACK_SCENARIOS[args.attack]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and encoder...")
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

    print("\n" + "=" * 60)
    print("  LIVE CYBERATTACK DETECTION DEMO")
    print(f"  Demo attack: {args.attack}")
    print(f"  Detector threshold: |z| >= {THRESHOLD_Z}")
    print("=" * 60 + "\n")

    cmd = ["sumo-gui", "-c", SUMOCFG, "--start"]
    traci.start(cmd)

    for _ in range(20):
        traci.simulationStep()

    # Pick ONE vehicle to track for the whole demo, and lock the camera on it -
    # otherwise the view stays zoomed out over the entire map (just green).
    tracked_veh_id = get_valid_vehicle()
    if tracked_veh_id is None:
        print("No vehicle found to track, exiting.")
        traci.close()
        return
    traci.gui.trackVehicle(VIEW_ID, tracked_veh_id)
    traci.gui.setZoom(VIEW_ID, 20000)
    print(f"Camera locked on vehicle: {tracked_veh_id}")

    attack_triggered = False
    near_stop_flag = 0
    check_count = 0

    print("Watching normal traffic...\n")
    time.sleep(2)

    try:
        while True:
            veh_id = tracked_veh_id if tracked_veh_id in traci.vehicle.getIDList() else None
            if veh_id is None:
                veh_id = get_valid_vehicle()
                if veh_id is None:
                    traci.simulationStep()
                    continue
                tracked_veh_id = veh_id
                traci.gui.trackVehicle(VIEW_ID, tracked_veh_id)
                traci.gui.setZoom(VIEW_ID, 20000)
                print(f"(vehicle left map, now tracking: {tracked_veh_id})")

            features_t = get_state_features(veh_id)
            if features_t is None:
                continue

            state_t_vec = build_state_vector(features_t, None, encoder, near_stop_flag)
            state_t = torch.tensor(state_t_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                raw_mean, _, _ = policy.forward(state_t)
                action = squash_to_action(torch.tanh(raw_mean), device)
                predicted_t1, risk_hat, _ = world_model(state_t, action)

            for _ in range(CHECK_INTERVAL):
                traci.simulationStep()

            check_id = veh_id if veh_id in traci.vehicle.getIDList() else get_valid_vehicle()
            features_t1 = get_state_features(check_id) if check_id else None
            if features_t1 is None:
                continue

            scenario_for_t1 = mapped_scenario if attack_triggered else None
            state_t1_vec = build_state_vector(features_t1, scenario_for_t1, encoder, near_stop_flag)
            actual_t1 = torch.tensor(state_t1_vec, dtype=torch.float32, device=device).unsqueeze(0)

            surprise = torch.nn.functional.mse_loss(predicted_t1[:, :6], actual_t1[:, :6]).item()
            z = abs(surprise - NORMAL_MEAN) / NORMAL_STD
            detector_says_attack = z >= THRESHOLD_Z

            timestamp = time.strftime("%H:%M:%S")
            ground_truth_label = f"ATTACK ACTIVE ({args.attack})" if attack_triggered else "NORMAL DRIVING"
            detector_label = ">>> ALERT: POSSIBLE ATTACK <<<" if detector_says_attack else "normal"
            correct = (detector_says_attack == attack_triggered)
            correctness_mark = "correct" if correct else "MISSED / FALSE ALARM"

            print(f"[{timestamp}] surprise={surprise:.4f}  z={z:.2f}")
            print(f"    REAL STATUS  : {ground_truth_label}")
            print(f"    DETECTOR SAYS: {detector_label}   ({correctness_mark})\n")

            check_count += 1
            if not attack_triggered and check_count >= NORMAL_PHASE_CHECKS:
                near_stop_flag, success = trigger_attack(args.attack, veh_id)
                if success:
                    attack_triggered = True

            time.sleep(PAUSE_BETWEEN_CHECKS)

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user (Ctrl+C). Closing SUMO...")
    except traci.exceptions.FatalTraCIError:
        print("\n\nSUMO window was closed. Demo ended.")
    finally:
        try:
            traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()