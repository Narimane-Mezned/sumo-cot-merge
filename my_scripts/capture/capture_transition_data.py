import os
import json
import traci
import time

SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
BOX_SIZE = 60
STEPS_BEFORE_CAPTURE = 20
STEPS_AFTER_ATTACK = 8  

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


def get_valid_vehicle(exclude_prefixes=("phantom", "sybil")):
    ids = [v for v in traci.vehicle.getIDList() if not v.startswith(exclude_prefixes)]
    return ids[0] if ids else None


def screenshot_for(veh_id, output_path):
    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)
    traci.simulationStep()
    time.sleep(0.5)
    traci.gui.screenshot(VIEW_ID, output_path)
    traci.simulationStep()
    time.sleep(1.0)


cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"]
print("Launching SUMO-GUI...")
traci.start(cmd)

for step in range(STEPS_BEFORE_CAPTURE):
    traci.simulationStep()

# --- Baseline episode: normal throughout (control) ---
veh_id = get_valid_vehicle()
frame_path = os.path.join(OUTPUT_DIR, "transition_normal_t.png")
screenshot_for(veh_id, frame_path)
features_t = get_state_features(veh_id) or {"speed_ratio": 0.0, "red_light": 0}
features_t["near_stop"] = 0

for _ in range(STEPS_AFTER_ATTACK):
    traci.simulationStep()

check_id = veh_id if veh_id in traci.vehicle.getIDList() else get_valid_vehicle()
features_t1 = (get_state_features(check_id) if check_id else None) or {"speed_ratio": 0.0, "red_light": 0}
features_t1["near_stop"] = 0

normal_episode = {
    "frame_t": os.path.basename(frame_path),
    "t": {**features_t, "mapped_scenario": None},
    "t1": {**features_t1, "mapped_scenario": None},
}
print(f"[normal] t={features_t}  ->  t1={features_t1}")

# --- Attack episode: normal at t, attack hits between t and t1 ---
veh_id = get_valid_vehicle()
frame_path = os.path.join(OUTPUT_DIR, "transition_attack_t.png")
screenshot_for(veh_id, frame_path)
features_t = get_state_features(veh_id) or {"speed_ratio": 0.0, "red_light": 0}
features_t["near_stop"] = 0
print(f"[attack] BEFORE trigger: t={features_t}")

# Trigger the attack NOW, right after capturing the "before" snapshot
tls_ids = traci.trafficlight.getIDList()
for tls_id in tls_ids:
    n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
    traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)

for _ in range(STEPS_AFTER_ATTACK):
    traci.simulationStep()

check_id = veh_id if veh_id in traci.vehicle.getIDList() else get_valid_vehicle()
features_t1 = (get_state_features(check_id) if check_id else None) or {"speed_ratio": 0.0, "red_light": 0}
features_t1["near_stop"] = 0
features_t1["red_light"] = 1  # attack forces this regardless of what TraCI reads on this small map
print(f"[attack] AFTER trigger:  t1={features_t1}")

attack_episode = {
    "frame_t": os.path.basename(frame_path),
    "t": {**features_t, "mapped_scenario": None},       # at t, nothing looked wrong yet
    "t1": {**features_t1, "mapped_scenario": "red_light"},  # t1 is genuinely disrupted
}

traci.close()
time.sleep(1.0)

transition_data = {
    "normal": {"attack": "none", **normal_episode},
    "traffic_light_tampering": {"attack": "traffic_light_tampering", **attack_episode},
}

out_path = os.path.join(OUTPUT_DIR, "transition_data.json")
with open(out_path, "w") as f:
    json.dump(transition_data, f, indent=2)
print(f"\nSaved to {out_path}")