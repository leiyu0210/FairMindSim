

# Emotional Dynamics in LLM Agents: Human-Like Behavior in Economic Games

This repository contains the code for experiments investigating the emotional dynamics of language model (LLM) agents in economic games. Our experiments are designed to simulate human-like behavior by analyzing the mood transitions of LLM agents in response to economic game scenarios.

## Prerequisites

Before you can run the experiments, you'll need to ensure your system meets the following requirements:

- Python 3.10
- pip

## Installation

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/leiyu0210/LLM4Psy.git
   cd LLM4Psy
   ```

2. Install the required Python packages:

   ```
   conda create -n llm_psy python=3.10
   conda activate llm_psy
   pip install -r requirements.txt
   ```

## Running the Experiment

To set up and run the experiments, use the following command:

```
python examples/agent_trust/all_game_person.py
```

## Workflow

The workflow of the experiments is as follows:

1. **Initialization**: An LLM agent is given a psychological scale to simulate the personality of the experimenter.

2. **Data Input**: The experiment begins by inputting real human experimental data to the agent.

3. **Mood Assessment**: The agent outputs its mood based on a predefined question table.

4. **Decision Making**: The agent completes the same mood table before making any decisions.

5. **Post-Decision Mood Assessment**: The mood table is completed once again after the decision-making process.

6. **Analysis**: The mood transitions are analyzed to understand how the agent's mood changes throughout the experiment.

7. **Output**: The mood transition tables are outputted using OpenAI's function calls.

## Contributions

We welcome contributions to improve the experiments or the analysis methods. If you're interested in contributing, please fork the repository and submit a pull request with your changes.

## Acknowledgements

Our code is based on the [Camel Project](https://github.com/camel-ai/camel). Special thanks to the Camel Project team for their foundational work that has significantly contributed to this project.

