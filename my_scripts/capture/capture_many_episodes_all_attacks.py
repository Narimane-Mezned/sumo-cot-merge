
import os
import json
import argparse
import random
import traci
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SUMOCFG = os.path.join(PROJECT_ROOT, "maps", "basic_simulation", "osm.sumocfg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_outputs")
VIEW_ID = "View #0"
BOX_SIZE = 60
MAX_WAIT_STEPS = 300
REAL_ROUTE_IDS = ["d_l", "d_u", "l_d", "u_d"] 
VTYPE = "car"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_CONDITIONS = [
    ("normal", None),
    ("traffic_light_tampering", "red_light"),
    ("universal_perturbation", "hard_brake"),
    ("sensor_spoofing", "hard_brake"),
    ("fake_safety", "near_stop"),
    ("fake_emergency", "near_stop"),
    ("sybil", "traffic_jam"),
]


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


def try_spawn_phantom(phantom_id, near_veh_id):
    
    try:
        x, y = traci.vehicle.getPosition(near_veh_id)
        edge_id = traci.vehicle.getRoadID(near_veh_id)
        route_id = random.choice(REAL_ROUTE_IDS)
        traci.vehicle.add(phantom_id, routeID=route_id, typeID=VTYPE, departPos="0", departLane="0")
        traci.vehicle.moveToXY(phantom_id, edge_id, 0, x + 6, y, keepRoute=2)
        return True
    except Exception as e:
        print(f"    (phantom spawn failed, continuing without it: {e})")
        return False


def apply_attack(attack_name, veh_id):
    
    near_stop_flag = 0

    if attack_name == "traffic_light_tampering":
        for tls_id in traci.trafficlight.getIDList():
            n_links = len(traci.trafficlight.getRedYellowGreenState(tls_id))
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)

    elif attack_name == "universal_perturbation":
       
        try:
            traci.vehicle.slowDown(veh_id, 0.0, 20.0) 
        except Exception:
            pass

    elif attack_name == "sensor_spoofing":
        try:
            traci.vehicle.setSpeed(veh_id, 0)  
        except Exception:
            pass

    elif attack_name == "fake_safety":
        near_stop_flag = 1
        try:
            lane_id = traci.vehicle.getLaneID(veh_id)
            pos_on_lane = traci.vehicle.getLanePosition(veh_id) + 15  # 15m ahead
            own_route_id = traci.vehicle.getRouteID(veh_id)  
            obstacle_id = f"phantom_obstacle_{random.randint(0,99999)}"
            traci.vehicle.add(vehID=obstacle_id, routeID=own_route_id, typeID=VTYPE)
            traci.vehicle.moveTo(obstacle_id, lane_id, pos_on_lane)
            traci.vehicle.setSpeed(obstacle_id, 0.0)
            traci.vehicle.setColor(obstacle_id, (255, 255, 0))
        except Exception as e:
            print(f"    (real-mechanism fake_safety spawn failed, falling back to direct enforcement: {e})")
            traci.vehicle.slowDown(veh_id, 0.5, 20.0)

    elif attack_name == "fake_emergency":
        try:
            traci.vehicle.slowDown(veh_id, 1.0, 3)  
            near_stop_flag = 1
        except Exception:
            pass

    elif attack_name == "sybil":
       
        try:
            max_speed = traci.vehicle.getMaxSpeed(veh_id)
            crawl_speed = max_speed * 0.15  
            traci.vehicle.slowDown(veh_id, crawl_speed, 20.0)  
        except Exception:
            pass
        
        try:
            x, y = traci.vehicle.getPosition(veh_id)
            edge_id = traci.vehicle.getRoadID(veh_id)
            for i in range(4):
                vid = f"sybil_{random.randint(0,99999)}_{i}"
                route_id = random.choice(REAL_ROUTE_IDS)
                traci.vehicle.add(vid, routeID=route_id, typeID=VTYPE, departPos="0", departLane="0")
                traci.vehicle.moveToXY(vid, edge_id, 0, x + (i * 7) - 12, y, keepRoute=2)
        except Exception as e:
            print(f"    (sybil ghost-vehicle spawn failed, ego enforcement still applied: {e})")

    return near_stop_flag


def run_one_episode(attack_name, mapped_scenario, episode_idx):
    seed = random.randint(1, 1_000_000)
    cmd = ["sumo-gui", "-c", SUMOCFG, "--start", "--quit-on-end", "--seed", str(seed)]
    traci.start(cmd)

    warmup_steps = random.randint(5, 60)
    for _ in range(warmup_steps):
        traci.simulationStep()

    veh_id = None
    for _ in range(MAX_WAIT_STEPS):
        traci.simulationStep()
        veh_id = get_valid_vehicle()
        if veh_id is not None:
            break
    if veh_id is None:
        traci.close()
        return None

    x, y = traci.vehicle.getPosition(veh_id)
    traci.gui.setBoundary(VIEW_ID, x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)
    traci.simulationStep()
    time.sleep(0.3)
    frame_name = f"{attack_name}_ep{episode_idx}_t.png"
    traci.gui.screenshot(VIEW_ID, os.path.join(OUTPUT_DIR, frame_name))
    traci.simulationStep()
    time.sleep(0.5)

    features_t = get_state_features(veh_id)
    if features_t is None:
        traci.close()
        return None
    features_t["near_stop"] = 0
    features_t["mapped_scenario"] = None

    near_stop_flag = 0
    if attack_name != "normal":
        near_stop_flag = apply_attack(attack_name, veh_id)

    steps_after = random.randint(5, 12)
    for _ in range(steps_after):
        traci.simulationStep()

    check_id = veh_id if veh_id in traci.vehicle.getIDList() else None
    if check_id is None:
      
        traci.close()
        return None
    features_t1 = get_state_features(check_id)
    if features_t1 is None:
        traci.close()
        return None
    features_t1["near_stop"] = near_stop_flag
    if attack_name != "normal":
        features_t1["mapped_scenario"] = mapped_scenario
        if mapped_scenario == "red_light":
            features_t1["red_light"] = 1
    else:
        features_t1["mapped_scenario"] = None

    traci.close()
    time.sleep(0.4)

    return {"frame_t": frame_name, "attack": attack_name, "mapped_scenario": mapped_scenario,
            "t": features_t, "t1": features_t1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-condition", type=int, default=8)
    args = parser.parse_args()

    all_episodes = {}
    for attack_name, mapped_scenario in ATTACK_CONDITIONS:
        print(f"\n=== {attack_name} ({args.episodes_per_condition} episodes) ===")
        for i in range(args.episodes_per_condition):
            print(f"  Episode {i+1}/{args.episodes_per_condition}...")
            result = run_one_episode(attack_name, mapped_scenario, i)
            if result:
                all_episodes[f"{attack_name}_ep{i}"] = result
                print(f"    t={result['t']} -> t1={result['t1']}")
            else:
                print(f"    Skipped.")

    out_path = os.path.join(OUTPUT_DIR, "all_attacks_data.json")
    with open(out_path, "w") as f:
        json.dump(all_episodes, f, indent=2)
    print(f"\nSaved {len(all_episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()