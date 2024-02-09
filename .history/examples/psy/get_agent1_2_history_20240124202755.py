import json
import numpy as np
from typing import Any, Dict

def generate_normal_values(mean: float, std_dev: float, lower_bound: float, upper_bound: float, count: int):
    """
    Generate a list of 'count' random values from a normal distribution with specified mean and standard deviation.
    Values are clipped to be within the specified bounds.
    """
    values = np.random.normal(mean, std_dev, count)
    values = np.clip(values, lower_bound, upper_bound)
    np.random.shuffle(values)  # Shuffle the values
    return values.tolist()

def fill_prompt_with_values(prompt: Dict[str, Any], key: str, values: list):
    """
    Fill the given prompt with the generated values at the specified key.
    Save the modified prompt as a JSON file.
    """
    modified_prompts = []
    for value in values:
        new_prompt = prompt.copy()
        new_prompt[key] = value
        modified_prompts.append(new_prompt)

    # Save the modified prompts to a JSON file
    with open('/mnt/data/modified_prompts.json', 'w') as file:
        json.dump(modified_prompts, file, indent=4)

    return "/mnt/data/modified_prompts.json"

# Example usage of the function
mean = 0.75
std_dev = 0.15
lower_bound = 0.3
upper_bound = 1.2
count = 20

# Generate 20 values
values = generate_normal_values(mean, std_dev, lower_bound, upper_bound, count)

# Example prompt
example_prompt = {"name": "example", "description": "This is an example.", "n": 0}

# Fill the prompt with generated values and save to a JSON file
json_file_path = fill_prompt_with_values(example_prompt, 'n', values)

json_file_path
