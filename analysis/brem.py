"""BREM (Belief-Reward Alignment Behavior Evolution Model) -- core implementation.

The paper's BREM is a dynamic feedback loop coupling an internal fairness
belief and an external reward through an *alignment potential*, with beliefs
updated by *dissonance-driven* (delta-rule) revision.

Paper formulation (faithful reference)
--------------------------------------
For trial j of agent i, with prior belief bel_{i,j-1}, norm-violation intensity
E_j, and accumulated external reward R_{i,j-1}:

    Phi_{i,j} = beta1 * bel_{i,j-1} * E_j + beta2 * R_{i,j-1} - C
    P(y_{i,j}=1) = sigmoid(Phi_{i,j} / T)             with T = 1 (fixed)
    delta_j = y_{i,j} - P(y_{i,j}=1)
    bel_{i,j} = bel_{i,j-1} + eta * delta_j           (no clipping)

R is rescaled by *dynamic Z-score normalization* in the paper's implementation
because E is bounded but R is not.

This file's deviations from the paper main model
------------------------------------------------
The class below ships a few optional extensions that we kept as ablation knobs.
They are all gated by parameters / flags and must be turned off to obtain the
paper-faithful BREM:

  1. Potential constant. Paper has a fixed scalar `-C`; this code uses
     `-beta_c * cost_t` (a per-trial cost). To recover the paper, pass
     `beta_c = 0` (and add a constant offset offline if you want a non-zero C).

  2. Temperature. Paper fixes `T = 1`. This code allows
     `T_t = T_base + lambda_T * f(arousal_t)` when `use_emotion=True`.
     For paper behavior pass `use_emotion=False` (default) and `T_base = 1.0`.

  3. Affect / arousal (AA, AC, EmoFDBK). Not in the paper's main model;
     active only when `use_emotion=True`. Disabled in parameter recovery.

  4. Belief clipping. Paper does not clip `bel`. This code clips to [0, 1]
     as a numerical safeguard; remove `np.clip(...)` for the paper-exact rule.

  5. Reward Z-score normalization on `wealth` is not yet applied here. To
     match the paper's implementation, replace `self.wealth` in the Phi
     computation with a running Z-score `(W - mean(W)) / std(W)`.

In short: ``BREMAgent(..., use_emotion=False)`` with `T_base=1.0`, `beta_c=0.0`,
no clipping, and Z-scored wealth gives the paper formulation. The defaults in
this file run the *extended* BREM used for ablations.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid


class BREMAgent:
    def __init__(self, agent_id, params, use_emotion=False):
        """
        :param agent_id: subject identifier
        :param params: dict of model parameters. ``beta1``, ``beta2``, ``eta``,
            ``bel_0`` are the paper-core params; ``beta_c``, ``T_base``,
            ``lambda_T`` are extensions (set ``beta_c=0``, ``T_base=1.0`` and
            ``use_emotion=False`` for paper behavior).
        :param use_emotion: if True, enables the affect / arousal extension and
            arousal-modulated temperature (NOT in the paper's main model).
        """
        self.id = agent_id
        self.params = params
        self.use_emotion = use_emotion

        self.belief = params['bel_0']
        self.wealth = 0.0
        self.history = []

    def step(self, trial_data):
        # 1. Parse environment input
        allocation = trial_data['amount_of_allocation']
        cost = trial_data['amount_of_cost']

        # Normalized unfairness signal (0 = fair, 1 = most unfair)
        E_j = (15.0 - allocation) / 5.0

        # ------------------------------------------------------
        # Phase 1: Appraisal & Temperature
        # ------------------------------------------------------
        # NOTE (deviation from paper): the paper has no Phase-1 affect /
        # temperature step -- it goes straight to the alignment potential
        # with `T = 1`. The block below is an optional ablation extension
        # that is active only when `use_emotion=True`. With the default
        # `use_emotion=False`, T stays at `params['T_base']` and AA is zero.
        aa_valence = 0.0
        aa_arousal = 0.0
        current_T = self.params['T_base']

        if self.use_emotion:
            # Anticipatory affect (AA): higher unfairness -> more negative
            # valence, higher arousal.
            noise_v = np.random.normal(0, 0.02)
            noise_a = np.random.normal(0, 0.02)

            aa_valence = 0.1 - (0.2 * E_j) + noise_v
            aa_arousal = -0.1 + (0.2 * E_j) + noise_a

            aa_valence = np.clip(aa_valence, -0.1, 0.1)
            aa_arousal = np.clip(aa_arousal, -0.1, 0.1)

            # Higher arousal -> higher T (more stochastic decisions).
            arousal_factor = (aa_arousal + 0.1) / 0.2  # normalize to 0~1
            current_T = self.params['T_base'] + (self.params['lambda_T'] * arousal_factor)

        current_T = max(current_T, 1e-3)

        # ------------------------------------------------------
        # Phase 2: Potential & Decision
        # ------------------------------------------------------
        # NOTE (deviation from paper): paper's Phi has a fixed scalar `-C`;
        # here we use `-beta_c * cost_t` so the term can vary across trials.
        # NOTE (deviation from paper): paper applies a dynamic Z-score
        # normalization to the running wealth `R`. Not applied here.
        # Set `beta_c = 0` (and Z-score `self.wealth`) to recover paper Phi.
        phi = (self.params['beta1'] * self.belief * E_j) + \
              (self.params['beta2'] * self.wealth) - \
              (self.params['beta_c'] * cost)

        prob_punish = expit(phi / current_T)
        choice = np.random.binomial(1, prob_punish)

        if choice == 1:
            self.wealth -= cost

        # ------------------------------------------------------
        # Phase 3: Dissonance & Emotional Feedback
        # ------------------------------------------------------
        dissonance = choice - prob_punish

        # NOTE (deviation from paper): paper's update is unconstrained;
        # we clip to [0, 1] purely for numerical stability.
        self.belief += self.params['eta'] * dissonance
        self.belief = np.clip(self.belief, 0.0, 1.0)

        ac_valence = 0.0
        ac_arousal = 0.0
        emo_fdbk_v = 0.0
        emo_fdbk_a = 0.0

        if self.use_emotion:
            # Arousal feedback: low |dissonance| (action matches belief) ->
            # arousal decreases; high |dissonance| -> arousal increases.
            emo_fdbk_a = (abs(dissonance) * 0.4) - 0.1

            # Valence feedback: choosing to punish yields retributive
            # satisfaction (positive); not punishing under high unfairness
            # yields guilt (negative).
            if choice == 1:
                emo_fdbk_v = 0.05 + (0.05 * E_j)
            else:
                emo_fdbk_v = -0.05 - (0.1 * E_j)

            ac_valence = np.clip(aa_valence + emo_fdbk_v, -0.1, 0.1)
            ac_arousal = np.clip(aa_arousal + emo_fdbk_a, -0.1, 0.1)

            # Recompute exact deltas after clipping.
            emo_fdbk_v = ac_valence - aa_valence
            emo_fdbk_a = ac_arousal - aa_arousal

        record = {
            'id': self.id,
            'trial': trial_data['trial'],
            'amount_of_allocation': allocation,
            'AA_valence': aa_valence if self.use_emotion else None,
            'AA_arousal': aa_arousal if self.use_emotion else None,
            'cost_level': 'high' if cost >= 5 else 'low',
            'amount_of_cost': cost,
            'choice': choice,
            'AC_valence': ac_valence if self.use_emotion else None,
            'AC_arousal': ac_arousal if self.use_emotion else None,
            'EmoFDBK_valence': emo_fdbk_v if self.use_emotion else None,
            'EmoFDBK_arousal': emo_fdbk_a if self.use_emotion else None,
            'internal_belief': self.belief,
            'internal_T': current_T,
            'internal_dissonance': dissonance,
        }
        self.history.append(record)
        return record


# ==========================================
# Synthetic-experiment driver
# ==========================================

def run_experiment(n_agents=10, emotion_mode=True):
    """Run a synthetic experiment with random allocations and costs."""
    params = {
        'beta1': 6.0,       # belief weight
        'beta2': 0.8,       # wealth weight
        'beta_c': 0.5,      # cost weight
        'T_base': 1.0,      # base temperature (rational noise floor)
        'lambda_T': 2.0,    # emotion coefficient: arousal -> T (only used if emotion_mode=True)
        'eta': 0.1,         # learning rate
        'bel_0': 0.0,       # initial belief
    }

    all_data = []

    for agent_id in range(1, n_agents + 1):
        agent = BREMAgent(agent_id, params, use_emotion=emotion_mode)

        # 20 random trials.
        allocations = np.random.choice([10, 11, 12, 13, 14, 15], 20)
        costs = np.random.choice(np.arange(0, 10), 20)

        for t in range(20):
            trial_input = {
                'trial': t + 1,
                'amount_of_allocation': allocations[t],
                'amount_of_cost': costs[t],
            }
            data_row = agent.step(trial_input)
            all_data.append(data_row)

    return pd.DataFrame(all_data)


# ==========================================
# Real-data simulation utilities
# ==========================================

def load_all_data(data_dir='data'):
    """Load all per-model and Human Excel files from `data_dir`.

    Files are expected to be named like
    ``result_persona_emotion_1.0_<agent_name>.xlsx``.
    """
    all_files = glob.glob(os.path.join(data_dir, '*.xlsx'))

    dfs_dict = {}

    for file_path in all_files:
        try:
            df = pd.read_excel(file_path)

            filename = os.path.basename(file_path)
            if 'Human' in filename:
                agent_name = 'Human'
            else:
                parts = filename.replace('.xlsx', '').split('_')
                try:
                    idx = parts.index('1.0')
                    agent_name = '_'.join(parts[idx + 1:])
                except ValueError:
                    agent_name = parts[-1]

            dfs_dict[agent_name] = df
            print(f"[OK] Loaded: {agent_name} ({len(df)} rows)")

        except Exception as e:
            print(f"[FAIL] Could not load {file_path}: {e}")

    return dfs_dict


def simulate_from_data(df, agent_name, emotion_mode=True, params=None):
    """Run BREM using allocations and costs taken from an existing dataset.

    :param df: real-data DataFrame
    :param agent_name: model / human label
    :param emotion_mode: whether to enable the emotion module
    :param params: BREM parameter dict (defaults if None)
    :return: simulation DataFrame
    """
    if params is None:
        params = {
            'beta1': 6.0,
            'beta2': 0.8,
            'beta_c': 0.5,
            'T_base': 1.0,
            'lambda_T': 2.0,
            'eta': 0.1,
            'bel_0': 0.5,
        }

    subject_ids = df['id'].unique()
    all_sim_data = []

    for subject_id in subject_ids:
        subject_data = df[df['id'] == subject_id].sort_values('trial')

        agent = BREMAgent(subject_id, params, use_emotion=emotion_mode)

        for _, row in subject_data.iterrows():
            trial_input = {
                'trial': row['trial'],
                'amount_of_allocation': row['amount_of_allocation'],
                'amount_of_cost': row['amount_of_cost'],
            }

            sim_record = agent.step(trial_input)
            sim_record['real_choice'] = row['choice']
            sim_record['agent_type'] = agent_name

            all_sim_data.append(sim_record)

    return pd.DataFrame(all_sim_data)


def compare_simulation_results(sim_df, real_df=None):
    """Summarize a simulation run and (optionally) compare against real data."""
    stats = {}

    stats['punishment_rate'] = sim_df['choice'].mean()

    if real_df is not None or 'real_choice' in sim_df.columns:
        if 'real_choice' in sim_df.columns:
            stats['accuracy'] = (sim_df['choice'] == sim_df['real_choice']).mean()
        else:
            merged = sim_df.merge(
                real_df[['id', 'trial', 'choice']],
                on=['id', 'trial'],
                suffixes=('_sim', '_real'),
            )
            stats['accuracy'] = (merged['choice_sim'] == merged['choice_real']).mean()

    if 'amount_of_allocation' in sim_df.columns:
        stats['punishment_by_allocation'] = (
            sim_df.groupby('amount_of_allocation')['choice'].mean().to_dict()
        )

    if 'cost_level' in sim_df.columns:
        stats['punishment_by_cost'] = sim_df.groupby('cost_level')['choice'].mean().to_dict()

    if sim_df['AA_valence'].notna().any():
        stats['emotion_stats'] = {
            'AA_valence_mean': sim_df['AA_valence'].mean(),
            'AA_arousal_mean': sim_df['AA_arousal'].mean(),
            'AC_valence_mean': sim_df['AC_valence'].mean(),
            'AC_arousal_mean': sim_df['AC_arousal'].mean(),
        }

    return stats


def run_full_simulation(data_dir='data', emotion_mode=True, save_results=True):
    """Run BREM over every Excel file in ``data_dir`` and report a summary."""
    print("=" * 60)
    print(f"BREM simulation - emotion mode: {'ON' if emotion_mode else 'OFF'}")
    print("=" * 60)

    print("\n[Step 1] Loading data ...")
    all_data = load_all_data(data_dir)

    if not all_data:
        print("No data files found.")
        return

    print(f"\n[Step 2] Simulating {len(all_data)} datasets ...")
    all_sim_results = {}
    all_stats = {}

    for agent_name, df in all_data.items():
        print(f"\nSimulating: {agent_name} ...")
        sim_df = simulate_from_data(df, agent_name, emotion_mode=emotion_mode)
        all_sim_results[agent_name] = sim_df

        stats = compare_simulation_results(sim_df)
        all_stats[agent_name] = stats

        print(f"  punishment rate: {stats['punishment_rate']:.3f}")
        if 'accuracy' in stats:
            print(f"  accuracy:        {stats['accuracy']:.3f}")

    print("\n" + "=" * 60)
    print("[Step 3] Summary")
    print("=" * 60)

    summary_data = []
    for agent_name, stats in all_stats.items():
        row = {
            'Agent': agent_name,
            'Punishment_Rate': stats['punishment_rate'],
            'Accuracy': stats.get('accuracy', np.nan),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data).sort_values('Punishment_Rate', ascending=False)

    print("\nPunishment rate and accuracy:")
    print(summary_df.to_string(index=False))

    if 'Human' in all_stats:
        print("\n" + "-" * 60)
        print("Human vs. AI models")
        print("-" * 60)
        human_rate = all_stats['Human']['punishment_rate']
        print(f"Human punishment rate: {human_rate:.3f}")
        print("\nDeviation from Human:")

        for agent_name, stats in all_stats.items():
            if agent_name != 'Human':
                diff = stats['punishment_rate'] - human_rate
                print(f"  {agent_name:40s}: {diff:+.3f}")

    if save_results:
        print("\n[Step 4] Saving results ...")
        output_dir = 'brem_simulation_results'
        os.makedirs(output_dir, exist_ok=True)

        mode_suffix = 'emotion' if emotion_mode else 'rational'

        summary_path = f"{output_dir}/summary_{mode_suffix}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  summary -> {summary_path}")

        for agent_name, sim_df in all_sim_results.items():
            safe_name = agent_name.replace('/', '_').replace(' ', '_')
            detail_path = f"{output_dir}/detail_{safe_name}_{mode_suffix}.csv"
            sim_df.to_csv(detail_path, index=False)
        print(f"  details -> {output_dir}/")

    print("\nDone.")

    return all_sim_results, all_stats, summary_df


if __name__ == "__main__":
    # Scenario 1: emotion module ON
    print("\n[Scenario 1] BREM-Emotion (emotion module ON)\n")
    results_emo, stats_emo, summary_emo = run_full_simulation(
        data_dir='data',
        emotion_mode=True,
        save_results=True,
    )

    print("\n" + "=" * 60 + "\n")

    # Scenario 2: emotion module OFF (purely rational)
    print("\n[Scenario 2] BREM-Rational (emotion module OFF)\n")
    results_rational, stats_rational, summary_rational = run_full_simulation(
        data_dir='data',
        emotion_mode=False,
        save_results=True,
    )

    print("\n" + "=" * 60)
    print("Emotion vs. Rational comparison")
    print("=" * 60)

    comparison = []
    for agent in summary_emo['Agent']:
        emo_rate = summary_emo[summary_emo['Agent'] == agent]['Punishment_Rate'].values[0]
        rat_rate = summary_rational[summary_rational['Agent'] == agent]['Punishment_Rate'].values[0]
        diff = emo_rate - rat_rate
        comparison.append({
            'Agent': agent,
            'Emotion_Mode': emo_rate,
            'Rational_Mode': rat_rate,
            'Difference': diff,
        })

    comparison_df = pd.DataFrame(comparison).sort_values('Difference', ascending=False)
    print("\nPer-model punishment-rate difference (Emotion - Rational):")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv('brem_simulation_results/mode_comparison.csv', index=False)
    print("\nComparison saved: brem_simulation_results/mode_comparison.csv")
