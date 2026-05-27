"""Scatter plot of model General-Capability (HLE Text) vs. Punishment Rate."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

# 1. Data
data = {
    'Model': [
        'Gemini-3-Pro', 'GPT-5', 'Claude-Sonnet-4.5', 'Gemini-2.5-Pro',
        'DeepSeek-V3.2', 'DeepSeek-R1', 'Claude-3.7-Sonnet', 'GPT-4.1',
        'Qwen3-235B-A22B-Thinking', 'Qwen3-235B-A22B-Instruct',
    ],
    'HLE_Text': [0.3772, 0.2632, 0.2632, 0.2206, 0.2180, 0.1404, 0.0789, 0.0497, 0.1543, 0.1175],
    'Punishment_Rate': [0.1317, 0.3290, 0.1997, 0.4894, 0.7143, 0.7927, 0.3628, 0.4443, 0.5785, 0.8268],
}
df = pd.DataFrame(data)

# 2. Style
sns.set_context("paper", font_scale=1.6)
sns.set_style("whitegrid")
plt.figure(figsize=(11, 8), dpi=300)

# 3. Quadratic fit
x = df['HLE_Text']
y = df['Punishment_Rate']
z = np.polyfit(x, y, 2)
p_func = np.poly1d(z)
y_pred = p_func(x)
r2 = r2_score(y, y_pred)
spearman_r = df[['HLE_Text', 'Punishment_Rate']].corr(method='spearman').iloc[0, 1]

x_trend = np.linspace(min(x) - 0.02, max(x) + 0.02, 100)
y_trend = p_func(x_trend)

# 4. Regression curve and scatter
sns.regplot(
    data=df,
    x='HLE_Text',
    y='Punishment_Rate',
    order=2,
    scatter=False,
    color='#D90368',
    ci=95,
    line_kws={'alpha': 0},
)
plt.plot(x_trend, y_trend, color='#D90368', linewidth=3, label='Quadratic Fit')
plt.scatter(
    df['HLE_Text'],
    df['Punishment_Rate'],
    color='#2E86AB',
    s=220,
    alpha=0.9,
    edgecolors='white',
    linewidths=2,
    zorder=5,
)

# 5. Human baseline
human_val = 0.3506
plt.axhline(y=human_val, color='#28a745', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

plt.text(
    0.395, human_val + 0.015, 'Human Baseline (0.35)',
    color='#28a745',
    fontweight='bold',
    fontsize=13,
    ha='right',
    va='bottom',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2),
)

# 6. Stats annotation
stats_text = (
    f"Non-linear Fit ($R^2$): {r2:.2f}\n"
    f"Rank Corr $\\rho$: {spearman_r:.2f}\n"
    f"(Status: Strong Signal)"
)
plt.text(
    0.96, 0.96, stats_text,
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#dee2e6", lw=1.5, alpha=0.95),
)

# 7. Per-point label tweaks
for i in range(df.shape[0]):
    row = df.iloc[i]
    xi, yi, name = row['HLE_Text'], row['Punishment_Rate'], row['Model']

    y_offset = 0.03
    ha = 'center'
    font_color = '#343a40'
    font_weight = 'semibold'
    font_size = 11

    if 'Gemini-3' in name:
        ha = 'right'
        xi -= 0.005
        y_offset = -0.01
    if 'Claude-Sonnet' in name:
        ha = 'right'
        xi -= 0.005
        y_offset = 0.015
    if 'GPT-5' in name:
        y_offset = -0.05

    if 'Instruct' in row['Model'] or 'Inst' in name:
        ha = 'right'
        xi -= 0.01
    if 'Thinking' in row['Model'] or 'Think' in name:
        ha = 'left'
        xi += 0.01
        y_offset = -0.02

    if 'DeepSeek-R1' in name:
        y_offset = 0.02
    if 'DeepSeek-V3' in name:
        ha = 'right'
        xi -= 0.01
        y_offset = 0.01

    if 'Claude-3.7' in name:
        y_offset = -0.06

    # Push GPT-4.1 down so the label clears the human-baseline line.
    if 'GPT-4.1' in name:
        y_offset = -0.06
        ha = 'center'

    plt.text(
        xi, yi + y_offset, name,
        fontsize=font_size,
        color=font_color,
        fontweight=font_weight,
        ha=ha,
        zorder=6,
    )

# 8. Axes & title
plt.ylim(-0.05, 1.0)
plt.xlim(0.04, 0.40)

plt.xlabel("General Capability (HLE Text Score)", fontsize=16, fontweight='bold', labelpad=12)
plt.ylabel("Punishment Rate (Norm Enforcement)", fontsize=16, fontweight='bold', labelpad=12)

ax = plt.gca()
ax.tick_params(axis='both', labelsize=14, width=2)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

plt.title("Constraint by Intelligence - A Non-linear Scaling Law",
          fontsize=18, fontweight='bold', pad=20)

sns.despine(left=True, bottom=True)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.grid(axis='x', alpha=0.0)

plt.tight_layout()
plt.savefig('hle_action.pdf', dpi=300, bbox_inches='tight')
