
import os
import traci
import time

SUMOCFG = "maps/basic_simulation/osm.sumocfg"
VIEW_ID = "View #0"
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_frame_zoomed.png")

STEPS_BEFORE_CAPTURE = 30
HALF_WIDTH = 30  

# --- Step 1: launch SUMO-GUI ---
cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end"]
print("Launching SUMO-GUI...")
traci.start(cmd)

# --- Step 2: run until at least one vehicle exists ---
vehicle_id = None
for step in range(STEPS_BEFORE_CAPTURE):
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    if vehicles:
        vehicle_id = vehicles[0]
        
        if step >= 5:
            break

if vehicle_id is None:
    print("WARNING: no vehicle found after", STEPS_BEFORE_CAPTURE, "steps.")
    print("Check that your .rou.xml / traffic demand actually spawns cars this early.")
    traci.close()
    raise SystemExit(1)

x, y = traci.vehicle.getPosition(vehicle_id)
print(f"Using vehicle '{vehicle_id}' at position ({x:.1f}, {y:.1f})")

# --- Step 3: set a tight camera boundary around that vehicle ---
traci.gui.setBoundary(
    VIEW_ID,
    x - HALF_WIDTH,
    y - HALF_WIDTH,
    x + HALF_WIDTH,
    y + HALF_WIDTH,
)
print(f"Camera boundary set to a {HALF_WIDTH*2}m x {HALF_WIDTH*2}m box around the vehicle.")

# Let the GUI actually redraw with the new camera position
traci.simulationStep()
time.sleep(1.0)

print(f"Requesting screenshot to {OUTPUT_PATH} ...")
traci.gui.screenshot(VIEW_ID, OUTPUT_PATH)
traci.simulationStep()
time.sleep(3.0)

traci.close()
time.sleep(1.0)

if os.path.exists(OUTPUT_PATH):
    size = os.path.getsize(OUTPUT_PATH)
    print(f"Done. File exists, size = {size} bytes.")
else:
    print("WARNING: file still not found.")

print(f"Full path: {os.path.abspath(OUTPUT_PATH)}")