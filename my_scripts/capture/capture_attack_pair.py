

import os
import json
import traci
import sumolib
import time

NET_FILE = "maps/basic_simulation/osm.net.xml"
SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
BOX_SIZE = 60  # meters around the vehicle
STEPS_BEFORE_CAPTURE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_zoomed(label, output_name):
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        print("No vehicles found yet.")
        return None
    veh_id = veh_ids[0]
    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(
        VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2,
        x + BOX_SIZE / 2, y + BOX_SIZE / 2,
    )
    traci.simulationStep()
    time.sleep(1.0)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    traci.gui.screenshot(VIEW_ID, output_path)
    traci.simulationStep()
    time.sleep(2.0)
    print(f"[{label}] Saved {output_path} (vehicle {veh_id} at {x:.1f},{y:.1f})")
    return output_name


cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"]
print("Launching SUMO-GUI...")
traci.start(cmd)

for step in range(STEPS_BEFORE_CAPTURE):
    traci.simulationStep()

ground_truth = {}

# --- Frame 1: normal, no attack ---
name1 = capture_zoomed("normal", "frame_normal.png")
if name1:
    ground_truth[name1] = "none"

# --- Trigger the attack: force every traffic light to red ---
tls_ids = traci.trafficlight.getIDList()
print(f"Traffic lights found: {tls_ids}")
for tls_id in tls_ids:
    n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
    traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)

for step in range(10):
    traci.simulationStep()

# --- Frame 2: attack active ---
name2 = capture_zoomed("attack: traffic_light_tampering", "frame_attack_traffic_light.png")
if name2:
    ground_truth[name2] = "traffic_light_tampering"

traci.close()
time.sleep(1.0)

gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Ground truth saved to {gt_path}: {ground_truth}")