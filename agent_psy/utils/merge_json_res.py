"""Merge a partial-run results JSON back into the main results JSON."""

import argparse
import json


def merge_json(subset_json_path, main_json_path, output_json_path):
    with open(subset_json_path, 'r') as f:
        subset_data = json.load(f)
    print(f"subset entries: {len(subset_data)}")

    with open(main_json_path, 'r') as f:
        main_data = json.load(f)
    print(f"main entries:   {len(main_data)}")

    # Overwrite main entries with values from the subset where keys match.
    for key, value in subset_data.items():
        if key in main_data:
            main_data[key] = value

    with open(output_json_path, 'w') as f:
        json.dump(main_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge a partial-run JSON into a main JSON (subset overrides main)."
    )
    parser.add_argument('subset_json', help="Partial-run / patch JSON.")
    parser.add_argument('main_json', help="Main JSON to merge into.")
    parser.add_argument('output_json', help="Output merged JSON path.")
    args = parser.parse_args()

    merge_json(args.subset_json, args.main_json, args.output_json)
