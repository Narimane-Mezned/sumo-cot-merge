import os
import json
import traci
import time

SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
BOX_SIZE = 60
STEPS_BEFORE_CAPTURE = 20
STEPS_BETWEEN_T_AND_T1 = 15

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
    """Pick any currently-alive real vehicle, freshly, right when needed."""
    ids = [v for v in traci.vehicle.getIDList() if not v.startswith(exclude_prefixes)]
    return ids[0] if ids else None


def capture_pair(label, output_prefix, near_stop_flag=0, force_red_light=None):
    veh_id = get_valid_vehicle()
    if veh_id is None:
        print(f"[{label}] No valid vehicle available, skipping.")
        return None

    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)
    traci.simulationStep()
    time.sleep(0.5)
    frame_t_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_t.png")
    traci.gui.screenshot(VIEW_ID, frame_t_path)
    traci.simulationStep()
    time.sleep(1.0)

    features_t = get_state_features(veh_id)
    if features_t is None:
        # vehicle left right after the screenshot - pick a fresh one for features
        veh_id = get_valid_vehicle()
        if veh_id is None:
            print(f"[{label}] Vehicle left before features could be read, skipping.")
            return None
        features_t = get_state_features(veh_id)
    features_t["near_stop"] = near_stop_flag
    if force_red_light is not None:
        features_t["red_light"] = force_red_light

    for _ in range(STEPS_BETWEEN_T_AND_T1):
        traci.simulationStep()

    features_t1 = get_state_features(veh_id)
    if features_t1 is None:
        
        fallback_id = get_valid_vehicle()
        if fallback_id is None:
            print(f"[{label}] No vehicle left for t+1 reading, skipping.")
            return None
        features_t1 = get_state_features(fallback_id)

    features_t1["near_stop"] = near_stop_flag
    if force_red_light is not None:
        features_t1["red_light"] = force_red_light

    print(f"[{label}] t={features_t}  ->  t+1={features_t1}")
    return {"frame_t": os.path.basename(frame_t_path), "t": features_t, "t1": features_t1}


cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"]
print("Launching SUMO-GUI...")
traci.start(cmd)

for step in range(STEPS_BEFORE_CAPTURE):
    traci.simulationStep()

surprise_data = {}

# --- Episode: normal ---
result = capture_pair("normal", "surprise_normal")
if result:
    surprise_data["normal"] = {"attack": "none", "mapped_scenario": None, **result}

# --- Episode: traffic_light_tampering -> red_light ---
tls_ids = traci.trafficlight.getIDList()
for tls_id in tls_ids:
    n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
    traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)
for _ in range(5):
    traci.simulationStep()

result = capture_pair("traffic_light_tampering", "surprise_traffic_light", force_red_light=1)
if result:
    surprise_data["traffic_light_tampering"] = {"attack": "traffic_light_tampering", "mapped_scenario": "red_light", **result}

traci.close()
time.sleep(1.0)

out_path = os.path.join(OUTPUT_DIR, "surprise_data.json")
with open(out_path, "w") as f:
    json.dump(surprise_data, f, indent=2)
print(f"\nSaved to {out_path}")