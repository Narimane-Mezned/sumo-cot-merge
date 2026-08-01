
import os
import json
import traci
import time

SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
BOX_SIZE = 60
STEPS_BEFORE_CAPTURE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_zoomed(output_name, target_veh_id=None):
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        print("No vehicles found.")
        return None
    veh_id = target_veh_id if (target_veh_id and target_veh_id in veh_ids) else veh_ids[0]
    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)
    traci.simulationStep()
    time.sleep(0.8)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    traci.gui.screenshot(VIEW_ID, output_path)
    traci.simulationStep()
    time.sleep(1.5)
    print(f"Saved {output_path} (vehicle {veh_id} at {x:.1f},{y:.1f})")
    return output_name


cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"]
print("Launching SUMO-GUI...")
traci.start(cmd)

for step in range(STEPS_BEFORE_CAPTURE):
    traci.simulationStep()

ground_truth = {}

# --- Episode 1: normal ---
name = capture_zoomed("ep1_normal.png")
if name:
    ground_truth[name] = "none"

# --- Episode 2: traffic_light_tampering ---
tls_ids = traci.trafficlight.getIDList()
for tls_id in tls_ids:
    n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
    traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)
for _ in range(10):
    traci.simulationStep()
name = capture_zoomed("ep2_traffic_light_tampering.png")
if name:
    ground_truth[name] = "traffic_light_tampering"

# --- Episode 3: fake_safety (phantom stopped obstacle in-lane) ---
real_ids = traci.vehicle.getIDList()
if real_ids:
    anchor_id = real_ids[0]
    ax, ay = traci.vehicle.getPosition(anchor_id)
    edge_id = traci.vehicle.getRoadID(anchor_id)
    try:
        traci.vehicle.add(
            "phantom_obstacle", routeID="", typeID="DEFAULT_VEHTYPE",
            departPos="0", departLane="0",
        )
        traci.vehicle.moveToXY("phantom_obstacle", edge_id, 0, ax + 5, ay, keepRoute=2)
        traci.vehicle.setSpeed("phantom_obstacle", 0)  # frozen in place = phantom blocker
        for _ in range(10):
            traci.simulationStep()
        name = capture_zoomed("ep3_fake_safety.png", target_veh_id=anchor_id)
        if name:
            ground_truth[name] = "fake_safety"
    except Exception as e:
        print(f"fake_safety injection failed, skipping: {e}")

# --- Episode 4: sybil (fake ghost vehicles inflating perceived traffic) ---
try:
    real_ids = traci.vehicle.getIDList()
    if real_ids:
        anchor_id = [v for v in real_ids if v != "phantom_obstacle"][0]
        ax, ay = traci.vehicle.getPosition(anchor_id)
        edge_id = traci.vehicle.getRoadID(anchor_id)
        for i in range(4):
            vid = f"sybil_{i}"
            traci.vehicle.add(vid, routeID="", typeID="DEFAULT_VEHTYPE", departPos="0", departLane="0")
            traci.vehicle.moveToXY(vid, edge_id, 0, ax + (i * 8) - 15, ay + 3, keepRoute=2)
        for _ in range(10):
            traci.simulationStep()
        name = capture_zoomed("ep4_sybil.png", target_veh_id=anchor_id)
        if name:
            ground_truth[name] = "sybil"
except Exception as e:
    print(f"sybil injection failed, skipping: {e}")

traci.close()
time.sleep(1.0)

gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"\nGround truth saved to {gt_path}:")
print(json.dumps(ground_truth, indent=2))