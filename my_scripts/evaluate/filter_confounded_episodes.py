import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    filtered = {}
    removed_by_condition = {}
    removed_no_compliance = {}

    for episode_name, entry in data.items():
        condition = entry.get("attack", "normal")
        already_red = entry.get("t", {}).get("red_light", 0) == 1

        if already_red:
            removed_by_condition[condition] = removed_by_condition.get(condition, 0) + 1
            continue

        
        if entry.get("mapped_scenario") == "red_light":
            if entry.get("t1", {}).get("red_light", 0) != 1:
                removed_no_compliance[condition] = removed_no_compliance.get(condition, 0) + 1
                continue

        filtered[episode_name] = entry

    print("Episodes removed (already at red light before the window started):")
    for condition, count in removed_by_condition.items():
        print(f"  {condition}: {count} removed")

    print("\nEpisodes removed (attack never reached the tracked vehicle):")
    for condition, count in removed_no_compliance.items():
        print(f"  {condition}: {count} removed")

    print(f"\nRemaining: {len(filtered)} / {len(data)} episodes")

    out_path = args.output or args.data.replace(".json", "_filtered.json")
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Saved filtered dataset to {out_path}")


if __name__ == "__main__":
    main()