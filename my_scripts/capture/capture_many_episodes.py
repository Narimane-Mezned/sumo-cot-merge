import os
import json
import argparse
import random
import traci
import time

SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
BOX_SIZE = 60
STEPS_BEFORE_CAPTURE = 20
STEPS_AFTER_ATTACK = 8
MAX_WAIT_STEPS = 300  
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    return ids[0] if ids else None


def run_one_episode(condition, episode_idx):
    
    seed = random.randint(1, 1_000_000)
    cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end", "--seed", str(seed)]
    traci.start(cmd)

    
    warmup_steps = random.randint(5, 60)
    for _ in range(warmup_steps):
        traci.simulationStep()

    veh_id = None
    for _ in range(MAX_WAIT_STEPS):
        traci.simulationStep()
        candidates = [v for v in traci.vehicle.getIDList() if not v.startswith(("phantom", "sybil"))]
        if candidates:
            veh_id = random.choice(candidates) 
            break

    if veh_id is None:
        traci.close()
        return None

    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)
    traci.simulationStep()
    time.sleep(0.3)
    frame_name = f"{condition}_ep{episode_idx}_t.png"
    traci.gui.screenshot(VIEW_ID, os.path.join(OUTPUT_DIR, frame_name))
    traci.simulationStep()
    time.sleep(0.6)

    features_t = get_state_features(veh_id)
    if features_t is None:
        traci.close()
        return None
    features_t["near_stop"] = 0
    features_t["mapped_scenario"] = None

    if condition == "attack":
        tls_ids = traci.trafficlight.getIDList()
        for tls_id in tls_ids:
            n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)

    steps_after = random.randint(5, 12)
    for _ in range(steps_after):
        traci.simulationStep()

    check_id = veh_id if veh_id in traci.vehicle.getIDList() else get_valid_vehicle()
    features_t1 = get_state_features(check_id) if check_id else None
    if features_t1 is None:
        traci.close()
        return None
    features_t1["near_stop"] = 0
    if condition == "attack":
        features_t1["red_light"] = 1
        features_t1["mapped_scenario"] = "red_light"
    else:
        features_t1["mapped_scenario"] = None

    traci.close()
    time.sleep(0.5)

    return {
        "frame_t": frame_name,
        "attack": "traffic_light_tampering" if condition == "attack" else "none",
        "t": features_t,
        "t1": features_t1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-condition", type=int, default=10)
    args = parser.parse_args()

    all_episodes = {}

    for condition in ["normal", "attack"]:
        print(f"\n=== Running {args.episodes_per_condition} episodes for: {condition} ===")
        for i in range(args.episodes_per_condition):
            print(f"  Episode {i+1}/{args.episodes_per_condition}...")
            result = run_one_episode(condition, i)
            if result:
                all_episodes[f"{condition}_ep{i}"] = result
                print(f"    t={result['t']} -> t1={result['t1']}")
            else:
                print(f"    Skipped (no vehicle appeared within {MAX_WAIT_STEPS} steps, or it left too fast).")

    out_path = os.path.join(OUTPUT_DIR, "multi_transition_data.json")
    with open(out_path, "w") as f:
        json.dump(all_episodes, f, indent=2)
    print(f"\nSaved {len(all_episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()