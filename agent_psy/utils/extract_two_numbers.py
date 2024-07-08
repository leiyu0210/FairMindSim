import json
import re

# Input JSON file name
input_file = "../gpt-4-1106-preview_res/check_chara_res/gpt-4-1106-preview_res/se_gpt-4-1106-preview.json"
# Output JSON file name
output_file = "se.json"


# Function to extract values using regular expressions
def extract_values(data):
    result = {}
    pattern_pleasure = re.compile(r"Pleasure-Displeasure item:\s*(-?\d+)")
    pattern_arousal = re.compile(r"Arousal-Sleepiness item:\s*(-?\d+)")

    for key, value in data.items():
        ans = value.get("ans", [])
        if len(ans) > 1:
            second_element = ans[1]
            pleasure_match = pattern_pleasure.search(second_element)
            arousal_match = pattern_arousal.search(second_element)

            if pleasure_match and arousal_match:
                pleasure_value = int(pleasure_match.group(1))
                arousal_value = int(arousal_match.group(1))

                result[key] = {
                    "Pleasure-Displeasure item": pleasure_value,
                    "Arousal-Sleepiness item": arousal_value,
                }

    return result


# Read the input JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Extract the required values
extracted_values = extract_values(data)

# Save the extracted values to the output JSON file
with open(output_file, "w") as f:
    json.dump(extracted_values, f, indent=4)

print(f"Extracted values have been saved to {output_file}")
