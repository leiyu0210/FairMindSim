"""Per-subject Human vs. model comparison: choice accuracy and emotion errors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EMOTION_FEATURES = [
    'AA_valence', 'AA_arousal',
    'AC_valence', 'AC_arousal',
    'EmoFDBK_valence', 'EmoFDBK_arousal',
]


def load_data(file_path):
    return pd.read_excel(file_path)


def calculate_individual_differences(human_data, model_data, subject_id):
    """Compute per-subject difference metrics between Human and one model."""
    human_subject = human_data[human_data['id'] == subject_id].copy()
    model_subject = model_data[model_data['id'] == subject_id].copy()

    if len(human_subject) == 0 or len(model_subject) == 0:
        return None

    # Sort by trial so rows correspond.
    human_subject = human_subject.sort_values('trial').reset_index(drop=True)
    model_subject = model_subject.sort_values('trial').reset_index(drop=True)

    if len(human_subject) != len(model_subject):
        print(
            f"Warning: subject {subject_id} has different trial counts: "
            f"Human={len(human_subject)}, Model={len(model_subject)}"
        )
        min_len = min(len(human_subject), len(model_subject))
        human_subject = human_subject.iloc[:min_len]
        model_subject = model_subject.iloc[:min_len]

    results = {}

    # Choice accuracy.
    if 'choice' in human_subject.columns and 'choice' in model_subject.columns:
        choice_match = (human_subject['choice'] == model_subject['choice']).sum()
        total_trials = len(human_subject)
        results['choice_accuracy'] = choice_match / total_trials if total_trials > 0 else 0
        results['choice_match_count'] = choice_match
        results['total_trials'] = total_trials

    # Emotion errors (MAE / RMSE / Pearson r).
    for feature in EMOTION_FEATURES:
        if feature in human_subject.columns and feature in model_subject.columns:
            mae = np.mean(np.abs(human_subject[feature] - model_subject[feature]))
            rmse = np.sqrt(np.mean((human_subject[feature] - model_subject[feature]) ** 2))
            corr = human_subject[feature].corr(model_subject[feature])

            results[f'{feature}_mae'] = mae
            results[f'{feature}_rmse'] = rmse
            results[f'{feature}_corr'] = corr

    return results


def compare_human_with_models(data_dir, output_dir):
    """Compare Human against every model in ``data_dir``."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Human data.
    human_file = data_dir / 'result_persona_emotion_1.0_Human.xlsx'
    print(f"Loading Human data from {human_file}")
    human_data = load_data(human_file)

    subject_ids = human_data['id'].unique()
    print(f"Found {len(subject_ids)} unique subjects")

    model_files = [
        f for f in data_dir.glob('result_persona_emotion_*.xlsx')
        if 'Human' not in f.name
    ]
    print(f"Found {len(model_files)} model files")

    all_results = []

    for model_file in model_files:
        model_name = model_file.stem.replace('result_persona_emotion_1.0_', '')
        print(f"\nProcessing model: {model_name}")

        model_data = load_data(model_file)

        individual_results = []
        for subject_id in subject_ids:
            diff_results = calculate_individual_differences(human_data, model_data, subject_id)

            if diff_results is not None:
                diff_results['subject_id'] = subject_id
                diff_results['model'] = model_name
                individual_results.append(diff_results)

        print(f"  Processed {len(individual_results)} subjects for {model_name}")
        all_results.extend(individual_results)

    results_df = pd.DataFrame(all_results)

    detailed_output = output_dir / 'individual_model_differences_detailed.csv'
    results_df.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to {detailed_output}")

    # Aggregate per model.
    aggregated_results = []
    for model_name in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_name]
        agg_result = {'model': model_name}

        numeric_cols = model_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            agg_result[f'{col}_mean'] = model_df[col].mean()
            agg_result[f'{col}_std'] = model_df[col].std()

        agg_result['num_subjects'] = len(model_df)
        aggregated_results.append(agg_result)

    aggregated_df = pd.DataFrame(aggregated_results)

    aggregated_output = output_dir / 'individual_model_differences_aggregated.csv'
    aggregated_df.to_csv(aggregated_output, index=False)
    print(f"Aggregated results saved to {aggregated_output}")

    # Compact summary.
    summary_results = []
    for model_name in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_name]

        summary = {
            'model': model_name,
            'num_subjects': len(model_df),
            'choice_accuracy_mean': model_df['choice_accuracy'].mean(),
            'choice_accuracy_std': model_df['choice_accuracy'].std(),
        }

        for feature in EMOTION_FEATURES:
            mae_col = f'{feature}_mae'
            if mae_col in model_df.columns:
                summary[f'{feature}_mae_mean'] = model_df[mae_col].mean()
                summary[f'{feature}_mae_std'] = model_df[mae_col].std()

            corr_col = f'{feature}_corr'
            if corr_col in model_df.columns:
                summary[f'{feature}_corr_mean'] = model_df[corr_col].mean()
                summary[f'{feature}_corr_std'] = model_df[corr_col].std()

        summary_results.append(summary)

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values('choice_accuracy_mean', ascending=False)

    summary_output = output_dir / 'individual_model_differences_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Summary results saved to {summary_output}")

    return results_df, aggregated_df, summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Per-subject Human vs. model comparison."
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help="Directory containing Human + per-model .xlsx files.",
    )
    parser.add_argument(
        '--output_dir',
        default='.',
        help="Directory to write per-subject comparison CSVs to.",
    )
    args = parser.parse_args()

    print("Starting per-subject Human vs. model comparison ...")
    print("=" * 80)

    results_df, aggregated_df, summary_df = compare_human_with_models(
        args.data_dir,
        args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("\nGenerated files:")
    print("  individual_model_differences_detailed.csv   - per-subject diffs")
    print("  individual_model_differences_aggregated.csv - per-model diffs (mean / std)")
    print("  individual_model_differences_summary.csv    - compact summary")

    print("\n" + "=" * 80)
    print("Choice accuracy (mean +/- std):")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:45s}: "
            f"{row['choice_accuracy_mean']:.4f} +/- {row['choice_accuracy_std']:.4f}"
        )
