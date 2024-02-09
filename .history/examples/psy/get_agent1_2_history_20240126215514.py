import json
from typing import Any, Dict

import numpy as np

mean = 0.75
std_dev = 0.15
lower_bound = 0.3
upper_bound = 1.2
count = 20


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
    modified_prompts = {}
    for i, value in enumerate(values):
        new_prompt = prompt.copy()
        modified_prompts[i] = new_prompt.format(v=value)
    return modified_prompts

def 

# Generate 20 values
values = generate_normal_values(mean, std_dev, lower_bound, upper_bound, count)

# Example prompt
example_prompt = {"name": "example",
                  "description": "This is an example.", "n": 0}

# Fill the prompt with generated values and save to a JSON file
json_file_path = fill_prompt_with_values(example_prompt, 'n', values)
