"""Behavioral analyses: punishment-rate alignment and emotional entropy."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'


# ==========================================
# 1. Data loading
# ==========================================

def load_all_data(data_dir):
    """Load every Human / model Excel file under ``data_dir``."""
    all_files = glob.glob(os.path.join(data_dir, '*.xlsx'))

    dfs = []
    model_names = []

    for file_path in all_files:
        try:
            df = pd.read_excel(file_path)

            filename = os.path.basename(file_path)
            if 'Human' in filename:
                agent_name = 'Human'
            else:
                # Extract model name, e.g.
                # 'result_persona_emotion_1.0_gpt-5-2025-08-07.xlsx' -> 'gpt-5-...'
                parts = filename.replace('.xlsx', '').split('_')
                try:
                    idx = parts.index('1.0')
                    agent_name = '_'.join(parts[idx + 1:])
                except ValueError:
                    agent_name = parts[-1]

            df['Agent Type'] = agent_name
            dfs.append(df)
            model_names.append(agent_name)
            print(f"[OK] Loaded: {agent_name} ({len(df)} rows)")

        except Exception as e:
            print(f"[FAIL] Could not load {file_path}: {e}")

    if not dfs:
        print("No valid data files found.")
        return None

    df_merged = pd.concat(dfs, ignore_index=True)

    if 'cost_level' in df_merged.columns:
        df_merged['cost_level'] = pd.Categorical(
            df_merged['cost_level'],
            categories=['low', 'high'],
            ordered=True,
        )

    print(f"\nTotal: {len(model_names)} datasets, {len(df_merged)} rows")
    print(f"Models: {', '.join(sorted(model_names))}")

    return df_merged


# ==========================================
# 2. Norm enforcement (action statistics)
# ==========================================

def analyze_action_patterns(df):
    print("\n" + "=" * 60)
    print("Norm Enforcement Patterns")
    print("=" * 60)

    # Overall punishment rate per agent.
    punishment_summary = (
        df.groupby('Agent Type')['choice']
        .agg(['mean', 'std', 'count'])
        .round(4)
    )
    punishment_summary.columns = ['Punishment Rate', 'Std Dev', 'N']
    print("\nOverall punishment rate:")
    print(punishment_summary.sort_values('Punishment Rate', ascending=False))

    if 'cost_level' in df.columns:
        detail_summary = (
            df.groupby(['Agent Type', 'amount_of_allocation', 'cost_level'], observed=True)['choice']
            .mean()
            .reset_index()
        )
        detail_summary.columns = ['Agent Type', 'Allocation', 'Cost Level', 'Punishment Rate']
        detail_summary.to_csv('summary_action_patterns.csv', index=False)
        print("\nDetailed statistics saved to: summary_action_patterns.csv")

    # Figure 1: punishment rate vs. allocation amount.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.lineplot(
        data=df,
        x='amount_of_allocation',
        y='choice',
        hue='Agent Type',
        markers=True,
        err_style="bars",
        errorbar=('ci', 95),
        ax=ax1,
    )
    ax1.set_title('Punishment Rate by Allocation Amount (All Models)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Amount of Allocation (Lower = More Unfair)', fontsize=12)
    ax1.set_ylabel('Punishment Rate', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    if 'cost_level' in df.columns:
        ax2 = axes[1]
        df_human = df[df['Agent Type'] == 'Human']
        sns.lineplot(
            data=df_human,
            x='amount_of_allocation',
            y='choice',
            hue='cost_level',
            markers=True,
            err_style="bars",
            errorbar=('ci', 95),
            ax=ax2,
            palette='Set2',
        )
        ax2.set_title('Human: Cost Level Effect', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Amount of Allocation', fontsize=12)
        ax2.set_ylabel('Punishment Rate', fontsize=12)
        ax2.legend(title='Cost Level', fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('fig_action_alignment_overview.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: fig_action_alignment_overview.png")
    plt.show()

    # Figure 2: heatmap of punishment rate per (Agent Type x allocation).
    pivot_data = (
        df.groupby(['Agent Type', 'amount_of_allocation'])['choice']
        .mean()
        .reset_index()
    )
    pivot_table = pivot_data.pivot(
        index='Agent Type',
        columns='amount_of_allocation',
        values='choice',
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Punishment Rate'},
        linewidths=0.5,
    )
    plt.title('Punishment Rate Heatmap: Agent Type x Allocation Amount', fontsize=14, fontweight='bold')
    plt.xlabel('Amount of Allocation', fontsize=12)
    plt.ylabel('Agent Type', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig_action_heatmap.png', dpi=300, bbox_inches='tight')
    print("Figure saved: fig_action_heatmap.png")
    plt.show()


# ==========================================
# 3. Emotional realism (entropy)
# ==========================================

def calculate_shannon_entropy(series, bins=20, range_limit=(-0.2, 0.2)):
    """Discretize a continuous series into ``bins`` and return Shannon entropy."""
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return np.nan

    clean_series = clean_series.clip(*range_limit)

    counts, _ = np.histogram(clean_series, bins=bins, range=range_limit, density=False)

    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # avoid log(0)

    if len(probs) == 0:
        return np.nan

    return entropy(probs, base=2)


def analyze_emotional_entropy(df):
    print("\n" + "=" * 60)
    print("Emotional Realism (Shannon entropy)")
    print("=" * 60)

    potential_emotion_cols = [
        'AA_valence', 'AA_arousal',
        'AC_valence', 'AC_arousal',
        'EmoFDBK_valence', 'EmoFDBK_arousal',
    ]
    emotion_cols = [col for col in potential_emotion_cols if col in df.columns]

    if not emotion_cols:
        print("Warning: no emotion columns found.")
        return

    print(f"\nEmotion metrics detected: {', '.join(emotion_cols)}")

    results = []

    for agent in sorted(df['Agent Type'].unique()):
        subset = df[df['Agent Type'] == agent]

        for col in emotion_cols:
            r_limit = (-0.2, 0.2) if 'EmoFDBK' in col else (-0.1, 0.1)

            if col not in subset.columns or subset[col].isna().all():
                print(f"  [skip] {agent} - {col}: no valid data")
                continue

            ent_val = calculate_shannon_entropy(subset[col], bins=20, range_limit=r_limit)
            mean_val = subset[col].mean()
            std_val = subset[col].std()

            results.append({
                'Agent Type': agent,
                'Emotion Metric': col,
                'Entropy': ent_val,
                'Mean': mean_val,
                'Std': std_val,
            })

    if not results:
        print("Warning: no entropy values could be computed.")
        return

    res_df = pd.DataFrame(results)

    res_df.to_csv('summary_emotion_entropy.csv', index=False)
    print("\nDetailed entropy values saved to: summary_emotion_entropy.csv")

    print("\nEmotional entropy summary:")
    print(res_df.to_string())

    avg_entropy = res_df.groupby('Agent Type')['Entropy'].mean().sort_values(ascending=False)
    print("\nMean emotional entropy per model (higher = closer to human variability):")
    print(avg_entropy.to_string())

    pivot_entropy = res_df.pivot(index='Agent Type', columns='Emotion Metric', values='Entropy')

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot_entropy,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Shannon Entropy (bits)'},
        linewidths=0.5,
    )
    plt.title('Emotional Entropy Heatmap: Higher = More Variability/Realism',
              fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Metric', fontsize=12)
    plt.ylabel('Agent Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fig_emotion_entropy_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: fig_emotion_entropy_heatmap.png")
    plt.show()

    n_metrics = len(emotion_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for idx, col in enumerate(emotion_cols):
        subset_df = res_df[res_df['Emotion Metric'] == col].sort_values('Entropy', ascending=False)

        ax = axes[idx]
        ax.bar(
            range(len(subset_df)),
            subset_df['Entropy'],
            color=['#e74c3c' if x == 'Human' else '#3498db' for x in subset_df['Agent Type']],
        )

        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Shannon Entropy (bits)', fontsize=10)
        ax.set_xticks(range(len(subset_df)))
        ax.set_xticklabels(subset_df['Agent Type'], rotation=90, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        if 'Human' in subset_df['Agent Type'].values:
            human_entropy = subset_df[subset_df['Agent Type'] == 'Human']['Entropy'].values[0]
            ax.axhline(y=human_entropy, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Human')

    plt.tight_layout()
    plt.savefig('fig_emotion_entropy_bars.png', dpi=300, bbox_inches='tight')
    print("Figure saved: fig_emotion_entropy_bars.png")
    plt.show()


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavioral and emotional alignment analyses for Human vs. LLM agents."
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help="Directory containing per-agent .xlsx files (default: ./data).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" Human vs. LLM behavioral / emotional alignment")
    print("=" * 70)

    print(f"\nLoading data from {args.data_dir} ...")
    df = load_all_data(args.data_dir)

    if df is None or len(df) == 0:
        print("No data loaded. Exiting.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("Data overview")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Agents: {df['Agent Type'].nunique()}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print("\nRow counts per agent:")
    print(df['Agent Type'].value_counts().sort_index())

    # Behavioral alignment.
    print("\n" + "=" * 70)
    print(" (1/2) Norm enforcement")
    print("=" * 70)
    analyze_action_patterns(df)

    # Emotional realism.
    print("\n" + "=" * 70)
    print(" (2/2) Emotional realism")
    print("=" * 70)
    analyze_emotional_entropy(df)

    print("\n" + "=" * 70)
    print(" Done.")
    print("=" * 70)
    print("\nGenerated files:")
    print("  fig_action_alignment_overview.png")
    print("  fig_action_heatmap.png")
    print("  fig_emotion_entropy_heatmap.png")
    print("  fig_emotion_entropy_bars.png")
    print("  summary_action_patterns.csv")
    print("  summary_emotion_entropy.csv")
    print("=" * 70)
