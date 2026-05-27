"""BREM forward simulation and per-subject parameter fitting.

This file fits the *cognitive-only* core of BREM (beta1, beta2, beta_c, eta)
to each subject's behavior by minimizing the negative log-likelihood under a
Bernoulli choice model with sigmoid policy. The emotion / temperature
extensions of `analysis/brem.py` are intentionally disabled here
(`use_emotion=False`, `T_base=1.0`, `lambda_T=0`) so that fitting recovers
the paper's cognitive parameters without confound.

Differences from the paper main model that remain in this file
--------------------------------------------------------------
1. Potential. The paper uses ``Phi = beta1*bel*E + beta2*R - C`` with a fixed
   scalar `C`. This file keeps a per-trial cost term ``- beta_c * cost_t`` as
   an additional ablation parameter. Set ``beta_c = 0`` (or remove the term)
   to recover the paper's Phi. We retain it because the third-party-punishment
   prompts in our LLM evaluation expose a per-trial `cost_of_punishment`, and
   we wanted to test whether LLMs are extra cost-sensitive relative to humans.

2. Reward Z-score normalization. The paper rescales the running wealth
   `R` with a dynamic Z-score because R is unbounded. This implementation
   does NOT yet apply the Z-score; if exact paper-parity matters for your
   downstream comparison, replace ``self.wealth`` with a running Z-scored
   wealth before the ``beta2 * wealth`` product.

3. Belief clipping. The paper update ``bel <- bel + eta * delta`` is
   unconstrained. The replay step here clips ``bel`` to [0, 1] purely for
   numerical stability. Remove the ``np.clip(...)`` line for the paper-exact
   delta rule.

The module-level ``BREMAgent`` and ``BREMFitter`` classes are deliberately
self-contained (kept separate from ``analysis/brem.py``) because fitting
turns off all the optional extensions, while ``analysis/brem.py`` is the
forward-simulation entry point that exposes the extensions as ablation knobs.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # Sigmoid


# ==========================================
# 1. Core model class (BREM Agent)
# ==========================================

class BREMAgent:
    def __init__(self, agent_id, params, use_emotion=False):
        self.id = agent_id
        self.params = params
        self.use_emotion = use_emotion
        self.belief = params.get('bel_0', 0)
        self.wealth = 0.0

    def step(self, trial_data, return_details=True):
        # --- Parse input ---
        allocation = trial_data['amount_of_allocation']
        cost = trial_data['amount_of_cost']
        E_j = (15.0 - allocation) / 5.0  # unfairness in [0, 1]

        # --- Phase 1: emotion & temperature ---
        current_T = self.params['T_base']
        if self.use_emotion:
            # Simple coupling: unfairness raises arousal, which raises T.
            aa_arousal = -0.1 + (0.2 * E_j)
            arousal_factor = (np.clip(aa_arousal, -0.1, 0.1) + 0.1) / 0.2
            current_T += (self.params.get('lambda_T', 0) * arousal_factor)
        current_T = max(current_T, 1e-3)

        # --- Phase 2: potential (Phi) ---
        # Intrinsic motivation term.
        phi_in = self.params['beta1'] * self.belief * E_j
        # Extrinsic motivation term (cost as immediate negative incentive).
        phi_ex = (self.params['beta2'] * self.wealth) - (self.params['beta_c'] * cost)

        phi = phi_in + phi_ex
        prob_punish = expit(phi / current_T)

        if not return_details:
            return prob_punish

        # --- Phase 3: decision replay & belief update ---
        # In fitting / analysis mode, force the recorded real choice so the
        # agent's belief evolves on the observed trajectory.
        choice = trial_data['real_choice']

        if choice == 1:
            self.wealth -= cost

        dissonance = choice - prob_punish

        old_belief = self.belief
        self.belief += self.params['eta'] * dissonance
        self.belief = np.clip(self.belief, 0.0, 1.0)

        record = {
            'trial': trial_data['trial'],
            'belief': old_belief,
            'dissonance': abs(dissonance),
            'intrinsic_force': abs(phi_in),
            'extrinsic_force': abs(phi_ex),
            'prob_punish': prob_punish,
            'choice': choice,
        }
        return record


# ==========================================
# 2. Per-subject fitter
# ==========================================

class BREMFitter:
    def __init__(self, df_agent):
        # Sort by trial.
        self.data = df_agent.sort_values('trial').reset_index(drop=True)

    def negative_log_likelihood(self, params_array):
        """Objective: minimize negative log-likelihood."""
        beta1, beta2, beta_c, eta = params_array

        # Hard constraints: parameters non-negative, learning rate in [0, 1].
        if eta < 0 or eta > 1 or beta1 < 0 or beta2 < 0 or beta_c < 0:
            return 1e10

        params = {
            'beta1': beta1,
            'beta2': beta2,
            'beta_c': beta_c,
            'eta': eta,
            'bel_0': 0.5,
            'T_base': 1.0,
            'lambda_T': 0,  # disable emotion during fitting (cognitive params only)
        }

        agent = BREMAgent(999, params, use_emotion=False)
        nll = 0.0

        for _, row in self.data.iterrows():
            trial_input = {
                'trial': row['trial'],
                'amount_of_allocation': row['amount_of_allocation'],
                'amount_of_cost': row['amount_of_cost'],
                'real_choice': row['choice'],
            }
            # return_details=True so the agent's belief is updated each step.
            rec = agent.step(trial_input, return_details=True)
            prob = np.clip(rec['prob_punish'], 1e-5, 1 - 1e-5)

            lik = np.log(prob) if row['choice'] == 1 else np.log(1 - prob)
            nll -= lik

        return nll

    def fit_and_analyze(self):
        """Fit and produce the per-subject metrics reported in the paper."""
        initial_guess = [5.0, 0.5, 0.5, 0.1]
        # Bounds: beta in [0, 20] / [0, 5], eta in [0.01, 1.0].
        bounds = ((0, 20), (0, 5), (0, 5), (0.01, 1.0))

        res = minimize(
            self.negative_log_likelihood,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
        )

        if not res.success:
            return None  # fitting failed

        best_params = {
            'beta1': res.x[0],
            'beta2': res.x[1],
            'beta_c': res.x[2],
            'eta': res.x[3],
            'bel_0': 0.5,
            'T_base': 1.0,
        }

        # Replay trajectory under the best parameters.
        agent = BREMAgent(999, best_params, use_emotion=False)

        intrinsic_forces = []
        extrinsic_forces = []
        dissonances = []
        beliefs = []

        for _, row in self.data.iterrows():
            inp = {
                'trial': row['trial'],
                'amount_of_allocation': row['amount_of_allocation'],
                'amount_of_cost': row['amount_of_cost'],
                'real_choice': row['choice'],
            }
            rec = agent.step(inp)

            intrinsic_forces.append(rec['intrinsic_force'])
            extrinsic_forces.append(rec['extrinsic_force'])
            dissonances.append(rec['dissonance'])
            beliefs.append(rec['belief'])

        mean_in = np.mean(intrinsic_forces)
        mean_ex = np.mean(extrinsic_forces)
        ratio = mean_in / (mean_ex + 1e-6)  # avoid div-by-zero

        return {
            'fitted_params': best_params,
            'motivation_ratio': ratio,
            'mean_dissonance': np.mean(dissonances),
            'belief_trajectory': beliefs,
            'nll': res.fun,
        }


# ==========================================
# 3. Batch processing of real datasets
# ==========================================

def run_analysis_on_real_data(data_dir='data'):
    files = glob.glob(os.path.join(data_dir, '*.xlsx'))

    if not files:
        print(f"Error: no .xlsx files in '{data_dir}'.")
        return

    all_results = []

    print(f"Found {len(files)} data files. Starting analysis ...\n")

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name} ...")

        try:
            df = pd.read_excel(file_path)

            required_cols = [
                'id', 'trial',
                'amount_of_allocation', 'amount_of_cost', 'choice',
            ]
            if not all(col in df.columns for col in required_cols):
                print(f"  [skip] missing columns in: {file_name}")
                continue

            subject_ids = df['id'].unique()

            for sub_id in subject_ids:
                sub_df = df[df['id'] == sub_id].copy()

                # Skip subjects with too few trials.
                if len(sub_df) < 10:
                    continue

                fitter = BREMFitter(sub_df)
                result = fitter.fit_and_analyze()

                if result:
                    row = {
                        'File': file_name,
                        'Subject_ID': sub_id,
                        'Motivation_Ratio': result['motivation_ratio'],
                        'Mean_Dissonance': result['mean_dissonance'],
                        'Beta1_Belief': result['fitted_params']['beta1'],
                        'BetaC_Cost': result['fitted_params']['beta_c'],
                        'Eta_Learning': result['fitted_params']['eta'],
                        'Final_Belief': result['belief_trajectory'][-1],
                        'Belief_Trajectory_Str': str(
                            np.round(result['belief_trajectory'], 2).tolist()
                        ),
                    }
                    all_results.append(row)
                    print(
                        f"  -> ID {sub_id}: "
                        f"Ratio={row['Motivation_Ratio']:.2f}, "
                        f"Dissonance={row['Mean_Dissonance']:.2f}"
                    )
                else:
                    print(f"  -> ID {sub_id}: fitting failed")

        except Exception as e:
            print(f"  [error] {e}")

    if all_results:
        result_df = pd.DataFrame(all_results)

        output_file = 'BREM_Analysis_Results.csv'
        result_df.to_csv(output_file, index=False)
        print("\n" + "=" * 50)
        print(f"Analysis complete. Results saved to: {output_file}")
        print("=" * 50)

        print("\nOverall mean metrics:")
        print(result_df[['Motivation_Ratio', 'Mean_Dissonance', 'Beta1_Belief']].mean())

        plot_sample_trajectory(result_df)

    else:
        print("\nNo results produced; please check the data format.")


def plot_sample_trajectory(df):
    """Plot belief trajectories of the highest and lowest Motivation Ratio agents."""
    try:
        df = df.sort_values('Motivation_Ratio', ascending=False)
        top_agent = df.iloc[0]
        bottom_agent = df.iloc[-1]

        traj_high = eval(top_agent['Belief_Trajectory_Str'])
        traj_low = eval(bottom_agent['Belief_Trajectory_Str'])

        plt.figure(figsize=(10, 5))
        plt.plot(
            traj_high,
            label=f"High Motivation Ratio ({top_agent['Motivation_Ratio']:.1f})",
            color='red',
        )
        plt.plot(
            traj_low,
            label=f"Low Motivation Ratio ({bottom_agent['Motivation_Ratio']:.1f})",
            color='blue',
            linestyle='--',
        )

        plt.title('Belief Evolution: High vs. Low Intrinsic Motivation Agent')
        plt.xlabel('Round (0-60)')
        plt.ylabel('Belief Strength (0-1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('belief_evolution_comparison_emotion.pdf')
        print("Sample trajectory figure saved.")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    run_analysis_on_real_data(data_dir='data')
