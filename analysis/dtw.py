"""Belief-evolution visualization & per-subject motivation analysis."""

import argparse
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ==========================================
# 1. Data loading
# ==========================================

def load_and_clean_data(csv_file):
    df = pd.read_csv(csv_file)

    def extract_model_name(filename):
        # Drop extensions and recover the trailing model name.
        name = filename.replace('.xlsx', '').replace('.csv', '')
        if 'emotion_1.0_' in name:
            return name.split('emotion_1.0_')[-1]
        elif '_' in name:
            return name.split('_')[-1]
        return name

    df['Model'] = df['File'].apply(extract_model_name)
    # Parse the stringified belief trajectory back into a real list.
    df['Belief_Trajectory'] = df['Belief_Trajectory_Str'].apply(ast.literal_eval)

    return df


# ==========================================
# 2. Belief-evolution comparison
# ==========================================

def load_belief_evolution_data(data_dir, mode='emotion'):
    """Load belief trajectories for every model under ``data_dir``.

    :param data_dir: directory containing files like ``detail_<model>_<mode>.csv``
    :param mode: ``'emotion'`` or ``'rational'``
    :return: dict mapping model name -> DataFrame with one trajectory per subject
    """
    detail_files = Path(data_dir).glob(f'detail_*_{mode}.csv')

    all_models_data = {}

    for file in detail_files:
        # filename: detail_<model>_<mode>.csv
        filename = file.stem
        model_name = filename.replace('detail_', '').replace(f'_{mode}', '')

        print(f"Loading: {model_name}")

        df = pd.read_csv(file)
        # One trajectory per subject id.
        grouped = df.groupby('id')['internal_belief'].apply(list).reset_index()

        all_models_data[model_name] = grouped

    return all_models_data


def plot_belief_evolution_comparison(
    data_dir='brem_simulation_results',
    mode='emotion',
    max_trials=60,
    save_path='plots/belief_evolution_comparison.png',
):
    """Plot mean belief trajectories (with std bands) for every model.

    :param data_dir: directory with ``detail_<model>_<mode>.csv`` files
    :param mode: ``'emotion'`` or ``'rational'``
    :param max_trials: maximum trial count to plot
    :param save_path: output figure path
    """
    all_models_data = load_belief_evolution_data(data_dir, mode)

    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_models_data)))

    for idx, (model_name, model_data) in enumerate(all_models_data.items()):
        belief_matrix = []

        for _, row in model_data.iterrows():
            trajectory = row['internal_belief']
            if len(trajectory) >= max_trials:
                belief_matrix.append(trajectory[:max_trials])

        if not belief_matrix:
            print(f"Warning: {model_name} has no trajectory long enough.")
            continue

        belief_matrix = np.array(belief_matrix)
        mean_belief = np.mean(belief_matrix, axis=0)
        std_belief = np.std(belief_matrix, axis=0)

        trials = np.arange(1, max_trials + 1)

        plt.plot(trials, mean_belief, label=model_name,
                 linewidth=2.5, alpha=0.8, color=colors[idx])
        plt.fill_between(trials,
                         mean_belief - std_belief,
                         mean_belief + std_belief,
                         alpha=0.15, color=colors[idx])

    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
                label='Initial Baseline (0.5)', alpha=0.6)

    plt.title(f'Belief Evolution Comparison: All Models (Mode: {mode.capitalize()})',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Trial Number', fontsize=14, fontweight='bold')
    plt.ylabel('Internal Belief (Mean +/- Std)', fontsize=14, fontweight='bold')

    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=11, framealpha=0.9, ncol=1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    plt.show()

    print("\n=== Sample sizes ===")
    for model_name, model_data in all_models_data.items():
        print(f"{model_name}: {len(model_data)} subjects")


# ==========================================
# 3. Per-subject motivation analyses (uses BREM_Analysis_Results.csv)
# ==========================================

def plot_analysis(df):
    sns.set(style="whitegrid", font_scale=1.1)

    # ------------------------------------------------------
    # Figure 1: per-model Motivation Ratio
    # ------------------------------------------------------
    plt.figure(figsize=(12, 6))

    df_sorted = df.sort_values('Motivation_Ratio', ascending=False)

    bar_plot = sns.barplot(x='Motivation_Ratio', y='Model', data=df_sorted, palette='viridis')

    plt.title('Intrinsic-Extrinsic Motivation Ratio by AI Model', fontsize=16, fontweight='bold')
    plt.xlabel('Motivation Ratio (Higher = More Driven by Justice/Belief)', fontsize=12)
    plt.ylabel('')

    for i, v in enumerate(df_sorted['Motivation_Ratio']):
        bar_plot.text(v + 0.05, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # Figure 2: cognitive dissonance vs. learning rate
    # ------------------------------------------------------
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x='Mean_Dissonance',
        y='Eta_Learning',
        hue='Model',
        style='Model',
        s=200,
        palette='deep',
    )

    plt.title('Cognitive Dissonance vs. Adaptability (Learning Rate)', fontsize=15)
    plt.xlabel('Mean Cognitive Dissonance (Conflict)', fontsize=12)
    plt.ylabel('Learning Rate (Eta)', fontsize=12)

    plt.axvline(x=df['Mean_Dissonance'].median(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=df['Eta_Learning'].median(), color='gray', linestyle='--', alpha=0.5)

    plt.text(df['Mean_Dissonance'].max() * 0.9, df['Eta_Learning'].max() * 0.9,
             'High Conflict\nFast Adapters', ha='right', color='red', alpha=0.6)
    plt.text(df['Mean_Dissonance'].min() * 1.1, df['Eta_Learning'].min() * 1.1,
             'Low Conflict\nSlow Adapters', ha='left', color='green', alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # Figure 3: belief evolution trajectories
    # ------------------------------------------------------
    plt.figure(figsize=(14, 7))

    for _, row in df.iterrows():
        traj = row['Belief_Trajectory']
        rounds = range(len(traj))

        # Line width scaled by motivation ratio.
        linewidth = 1.5 + (row['Motivation_Ratio'] / df['Motivation_Ratio'].max()) * 2.5
        plt.plot(rounds, traj, label=row['Model'], linewidth=linewidth, alpha=0.8)

    plt.title('Evolution of Justice Belief Over 60 Rounds', fontsize=16)
    plt.xlabel('Round (Trial)', fontsize=12)
    plt.ylabel('Belief Strength (0=Uncertain/Weak, 1=Strong)', fontsize=12)
    plt.ylim(0, 1.05)

    plt.axhline(y=0.5, color='gray', linestyle='--', label='Initial Baseline')

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # Figure 4: Beta1 (idealism) vs. BetaC (pragmatism)
    # ------------------------------------------------------
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x='BetaC_Cost',
        y='Beta1_Belief',
        hue='Model',
        s=300,
        edgecolor='black',
        alpha=0.8,
    )

    plt.title('Intrinsic Idealism (Beta1) vs. Extrinsic Pragmatism (BetaC)', fontsize=15)
    plt.xlabel('Sensitivity to Cost (Pragmatism)', fontsize=12)
    plt.ylabel('Weight on Belief/Fairness (Idealism)', fontsize=12)

    for i in range(df.shape[0]):
        plt.text(
            df.BetaC_Cost.iloc[i] + 0.02,
            df.Beta1_Belief.iloc[i],
            df.Model.iloc[i],
            fontsize=9,
            alpha=0.7,
        )

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. CLI
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief-evolution visualization.")
    parser.add_argument(
        '--data_dir',
        default='brem_simulation_results',
        help="Directory with detail_<model>_<mode>.csv files.",
    )
    parser.add_argument(
        '--mode',
        default='emotion',
        choices=['emotion', 'rational'],
        help="Which BREM mode to plot.",
    )
    parser.add_argument(
        '--max_trials',
        type=int,
        default=60,
        help="Maximum trial count to plot.",
    )
    parser.add_argument(
        '--save_path',
        default='plots/belief_evolution_comparison_emotion.png',
        help="Output figure path.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Belief evolution comparison")
    print("=" * 60)

    plot_belief_evolution_comparison(
        data_dir=args.data_dir,
        mode=args.mode,
        max_trials=args.max_trials,
        save_path=args.save_path,
    )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
