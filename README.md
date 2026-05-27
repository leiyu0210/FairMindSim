# Are LLMs Socially Adaptive? Contrasting Belief Evolution in Large Language Models and Humans

This repository contains the official code for our **KDD 2026** paper:

> **Are LLMs Socially Adaptive? Contrasting Belief Evolution in Large Language Models and Humans**

We study whether Large Language Model (LLM) agents update their beliefs and adapt their behavior in social/economic dilemmas the way humans do. We pair (i) a human-grounded behavioral simulation built on a third-party punishment game with (ii) **BREM (Belief-Reward Alignment Behavior Evolution Model)**, a cognitively motivated computational model that treats each trial as a dynamic feedback loop between an internal fairness belief and external reward, coupled through an *alignment potential* and updated by *dissonance-driven belief revision*. Comparing humans against a wide range of contemporary LLMs (GPT-5, GPT-4.1, Claude Sonnet 4.5, Claude 3.7 Sonnet, Gemini 2.5/3 Pro, DeepSeek R1/V3.2, Qwen3-235B Instruct/Thinking) reveals systematic gaps in how LLM agents revise beliefs and respond to perceived unfairness.

---

## Repository Structure

```
FairMindSim/
├── agent_psy/                    # Data-collection pipeline (LLM-as-participant simulation)
│   ├── all_game_person.py        # Main entry point for the third-party punishment game
│   ├── all_game_person_addition. # Variant for additional persona pool
│   ├── exp_model_class.py        # Model registry / API wrappers
│   ├── function_calls.py         # Function-calling schemas for structured emotion output
│   ├── structure_output.py       # Structured output post-processing
│   ├── utils/                    # Formatting and parsing helpers
│   │   ├── format_agent.py
│   │   ├── format_output.py
│   │   ├── extract_two_numbers.py
│   │   └── merge_json_res.py
│   └── data/                     # Experimental design only (allocations, scales, prompts)
│       ├── allocation_*.json     # Allocation schedules per condition
│       ├── aq_questions_list.json
│       ├── sds_questions_list.json
│       ├── game_prompt.py        # PANAS-variant prompts
│       ├── game_prompt_co.py     # Valence/arousal-variant prompts
│       ├── data_process.py       # Persona assembly from AQ/SDS responses
│       └── design_exp.py         # Allocation-schedule generator
│       # NOTE: raw human / model trial data are NOT shipped — see "Data Availability".
└── analysis/                     # BREM modeling, fitting, and analysis
    ├── brem.py                   # Core BREM simulation (emotion-on / emotion-off)
    ├── forward.py                # Per-subject fitter (negative-log-likelihood)
    ├── analysis.py               # Punishment rate & emotional-entropy analyses
    ├── model_comparison.py       # Per-subject Human vs. model comparison
    ├── dtw.py                    # Belief-trajectory comparison
    └── vis.py                    # HLE-vs-Punishment-Rate scaling figure
```

---

## Prerequisites

- Python 3.10
- pip / conda

## Installation

```bash
git clone https://github.com/leiyu0210/FairMindSim.git
cd FairMindSim

conda create -n fairmindsim python=3.10
conda activate fairmindsim
pip install -r requirements.txt
```

Set the API keys for whichever providers you intend to query before launching the
data-collection pipeline:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
# ... etc.
```

---

## Part 1 · Data Collection (`agent_psy/`)

`agent_psy/` implements the LLM-as-participant pipeline used to elicit behavior and emotion trajectories from both human-grounded personas and LLM agents in a 20-trial third-party punishment game.

### Workflow

1. **Persona initialization.** Each agent is conditioned on a profile derived from validated psychological scales (AQ, SDS) plus demographic attributes (age, gender). Scale items live in [agent_psy/data/aq_questions_list.json](agent_psy/data/aq_questions_list.json) and [agent_psy/data/sds_questions_list.json](agent_psy/data/sds_questions_list.json); persona assembly is in [agent_psy/data/data_process.py](agent_psy/data/data_process.py).
2. **Game setup.** The third-party punishment game and its prompts are defined in [agent_psy/data/game_prompt.py](agent_psy/data/game_prompt.py) (PANAS variant) and [agent_psy/data/game_prompt_co.py](agent_psy/data/game_prompt_co.py) (valence/arousal variant). Allocation schedules per condition live in [agent_psy/data/allocation_se.json](agent_psy/data/allocation_se.json), [agent_psy/data/allocation_se_co.json](agent_psy/data/allocation_se_co.json), and [agent_psy/data/allocation_ex.json](agent_psy/data/allocation_ex.json).
3. **Trial loop (per agent, 20 rounds).** For each round, the agent (a) reports its emotion before judging, (b) issues an Accept/Reject judgment, and (c) reports its emotion after judging. Structured outputs are enforced via OpenAI-style function calling — see [agent_psy/function_calls.py](agent_psy/function_calls.py) and [agent_psy/utils/format_output.py](agent_psy/utils/format_output.py).
4. **Multi-model dispatch.** [agent_psy/all_game_person.py](agent_psy/all_game_person.py) runs the loop across the model list registered in [agent_psy/exp_model_class.py](agent_psy/exp_model_class.py) (OpenAI, Anthropic, Google, DeepSeek, Qwen, ChatGLM, and open-source backends via Camel/vLLM).
5. **Post-processing.** [agent_psy/utils/format_agent.py](agent_psy/utils/format_agent.py) and [agent_psy/utils/format_output.py](agent_psy/utils/format_output.py) parse the raw responses into per-trial records used downstream by BREM.

### Running the simulation

```bash
cd agent_psy
python all_game_person.py
```

The entrypoint loads experiment-design and persona files lazily inside `__main__`, so the module is importable without the data — but launching it requires the data files described in **Data Availability**.

### Data Availability

The raw human survey responses, the constructed character profiles, and the per-model trial-level outputs are **not included** in this repository for privacy and IRB reasons. Only the experimental design files (allocation schedules, scale items, prompts) are kept under [agent_psy/data/](agent_psy/data/).

If you need access to the data for **non-commercial academic research**, please email:

**liu-h21@mails.tsinghua.edu.cn**

Please include in your email: (1) your name and affiliation, (2) the intended use, and (3) confirmation that the data will not be redistributed.

---

## Part 2 · BREM: Belief-Reward Alignment Behavior Evolution Model (`analysis/`)

**BREM (Belief-Reward Alignment Behavior Evolution Model)** is the computational model proposed in the paper. It treats each trial as a dynamic feedback loop between an *internal fairness belief* and an *external reward / wealth* signal, coupled through an **alignment potential** and updated by **dissonance-driven belief revision**. Fitting BREM per subject (human or LLM agent) lets us directly compare *parameters* and *dynamics* of belief evolution across populations, rather than relying on static behavioral classifications.

### Method overview (paper formulation)

For trial $j$, let $bel_{i,j-1}$ be agent $i$'s prior fairness belief, $E_j$ the **norm-violation intensity** of the observed allocation, and $R_{i,j-1}$ the agent's accumulated external reward.

**Phase 1 — Alignment potential.** The internal-vs-external alignment potential is

$$
\Phi_{i,j} \;=\; \beta_1 \cdot bel_{i,j-1} \cdot E_j \;+\; \beta_2 \cdot R_{i,j-1} \;-\; C,
$$

where $C$ is a fixed cognitive-cost / threshold constant. Because $E_j$ is bounded but $R$ is unbounded, $R$ is rescaled by **dynamic Z-score normalization** in the implementation to keep the two driving terms on comparable scales.

**Phase 2 — Boltzmann-like policy.** The binary punishment decision is sampled from

$$
P(y_{i,j} = 1) \;=\; \sigma(\Phi_{i,j} / T), \qquad y_{i,j} \sim \text{Bernoulli}\big(P(y_{i,j}=1)\big).
$$

We fix $T = 1$ throughout this work (no learned temperature).

**Phase 3 — Dissonance-driven belief update.** Define the prediction error / dissonance

$$
\delta_j \;=\; y_{i,j} - P(y_{i,j}=1),
$$

and update the belief by

$$
bel_{i,j} \;\leftarrow\; bel_{i,j-1} + \eta\, \delta_j.
$$

This is exactly the delta rule arising from stochastic-gradient ascent on the Bernoulli log-likelihood; the full derivation is given in the paper.

This decomposition lets us ask: *do LLMs update beliefs as quickly as humans?* ($\eta$), *do they weight unfairness vs. reward the same way?* ($\beta_1 / \beta_2$), and *do they exhibit the same alignment-potential dynamics over trials?* ($\Phi$ trajectories).

### Implementation notes (deviations from the paper)

The released [analysis/brem.py](analysis/brem.py) ships with a few **optional extensions** beyond the paper's main-text formulation. They are gated by flags / parameters and must be turned off to obtain the paper-faithful BREM. We list each gap explicitly so that the README and the code stay honest about the difference.

| Component | Paper main model | Code default | How to recover paper behavior |
|---|---|---|---|
| Potential constant | $-C$ (fixed scalar) | $-\beta_c \cdot c_t$ (per-trial cost) | Set `beta_c = 0` and add a constant offset; the per-trial cost is an extension we kept to study cost-sensitivity. |
| Temperature | $T = 1$ fixed | $T_t = T_{\text{base}} + \lambda_T \cdot f(\alpha^{\text{AA}}_t)$ when `use_emotion=True` | Pass `use_emotion=False` (the default) and `T_base=1.0`. |
| Affect / arousal module | not in the main model | available when `use_emotion=True` | Pass `use_emotion=False`; this is the configuration used for parameter recovery. |
| Belief clipping | $bel$ unconstrained | $\mathrm{clip}(bel_t + \eta\delta_t, 0, 1)$ | Numerical safeguard only; remove the `np.clip(...)` line for the paper-exact rule. |
| Reward Z-score | dynamic Z-score on $R$ | not yet applied in the simulation path | Apply running Z-score `(W - W.mean()) / W.std()` on the wealth before the $\beta_2 R$ term. |

The cost term ($-\beta_c c_t$), the affect / arousal module, and the temperature modulation are **not in the paper's main BREM model**. They remain in the code as ablation knobs that we (and other researchers) can use to probe whether emotion-modulated stochasticity, cost-sensitivity, or per-trial pricing reproduces human-like response variability. To run paper-faithful BREM, instantiate `BREMAgent(..., use_emotion=False)` with `T_base=1.0` and `beta_c=0.0`, and follow the in-file notes in [analysis/brem.py](analysis/brem.py) for re-introducing dynamic Z-score normalization on $R$.


## Acknowledgements

The simulation pipeline builds on the [Camel Project](https://github.com/camel-ai/camel). We thank the Camel team for their foundational work.

