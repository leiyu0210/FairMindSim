# Total summary of the code to generate the final chart with custom model colors and matching baseline colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


human_file_path = './data/human_data/human_data_all_0303.xlsx'
gpt3_5_file_path = './all_agents_gpt3_volatility_mean.xlsx'
gpt4_file_path = './all_agents_gpt4_volatility_mean.xlsx'

human_data = pd.read_excel(human_file_path)
gpt3_5_data = pd.read_excel(gpt3_5_file_path)
gpt4_data = pd.read_excel(gpt4_file_path)


human_rename_dict = {'id': 'ID', 'MvEmo': 'current_valence',
                     'MvEp': 'anticipated_valence', 'MvRe': 'actual_valence'}
human_data.rename(columns=human_rename_dict, inplace=True)
emotion_indicators = ['current_valence',
                      'anticipated_valence', 'actual_valence']

human_data_standardized = standardize(human_data.copy(), emotion_indicators)
gpt3_5_data_standardized = standardize(gpt3_5_data.copy(), emotion_indicators)
gpt4_data_standardized = standardize(gpt4_data.copy(), emotion_indicators)
age_grouped_stats = human_data_standardized.groupby('group')['age'].describe()
gender_grouped_distribution = human_data_standardized.groupby('group')[
    'gender'].value_counts()

human_data_standardized['Source'] = 'Human'
gpt3_5_data_standardized['Source'] = 'GPT-3.5'
gpt4_data_standardized['Source'] = 'GPT-4'
merged_data = human_data_standardized.merge(
    gpt3_5_data_standardized, on='ID', how='outer', suffixes=('', '_gpt3_5'))
merged_data = merged_data.merge(
    gpt4_data_standardized, on='ID', how='outer', suffixes=('', '_gpt4'))

print(merged_data.columns)

baseline_human = {
    'current_valence': merged_data['current_valence'].mean(),
    'anticipated_valence': merged_data['anticipated_valence'].mean(),
    'actual_valence': merged_data['actual_valence'].mean()
}
baseline_gpt3_5 = {
    'current_valence': merged_data['current_valence_gpt3_5'].mean(),
    'anticipated_valence': merged_data['anticipated_valence_gpt3_5'].mean(),
    'actual_valence': merged_data['actual_valence_gpt3_5'].mean()
}
baseline_gpt4 = {
    'current_valence': merged_data['current_valence_gpt4'].mean(),
    'anticipated_valence': merged_data['anticipated_valence_gpt4'].mean(),
    'actual_valence': merged_data['actual_valence_gpt4'].mean()
}
baseline_scores = {
    'Human': np.mean(list(baseline_human.values())),
    'GPT-3.5': np.mean(list(baseline_gpt3_5.values())),
    'GPT-4': np.mean(list(baseline_gpt4.values()))
}


# Calculate deviations for each model in each category
deviations = {

}

renamed_categories = ['All', 'Female', 'Male']

color_dict = {
    "GPT-3.5": "red",
    "GPT-4": "blue",
    "Human": "purple"
}
color_alpha_dict = {
    "GPT-3.5": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
        0.7,
    ),
    "GPT-4": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 0.7),
    "Human": (
        0.8901960784313725,
        0.4666666666666667,
        0.7607843137254902,
        0.7,
    )
}
models = ['Human', 'GPT-3.5', 'GPT-4']
custom_colors = [color_alpha_dict.get(model[:-4], "gray") for model in models]

categories_group_1 = renamed_categories
categories_group_2 = renamed_categories
categories_group_3 = renamed_categories


def draw_brace(
    ax,
    xspan,
    text,
    y_offset=0,
    y_height=0.05,
    color="black",
    fontsize=20,
    draw_brace=True,
):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin

    brace_id, brace_annotation, text_id, text_annotation = None, None, None, None

    # If draw_brace is True, then draw the brace
    if draw_brace:
        # Define the brace shape
        brace_id = "{0}-{1}-brace".format(xmin, xmax)
        brace_annotation = ax.annotate(
            "",
            xy=(xmax, ymin - y_offset),
            xytext=(xmin, ymin - y_offset),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<->",  # Changed from '<->' to 'brace' which is the correct style for a brace
                lw=2,
                color=color,
            ),
            annotation_clip=False,
        )

    # Add text annotation based on the brace or just at the position if brace is not drawn
    text_id = "{0}-{1}-text".format(xmin, xmax)
    text_annotation = ax.annotate(
        text,
        xy=(
            np.mean([xmin, xmax]),
            ymin - y_offset - (yspan * y_height *
                               1.5 if draw_brace else 0) - 0.6,
        ),
        xytext=(0, -10),
        textcoords="offset points",
        xycoords="data",
        ha="center",
        va="top",
        color=color,
        fontsize=fontsize,
        annotation_clip=False,
    )

    return brace_id, brace_annotation, text_id, text_annotation


# 所有类别
categories = categories_group_1 + categories_group_2 + categories_group_3
# Plotting parameters
bar_width = 0.1
offsets = np.linspace(
    -bar_width * len(models) / 2, bar_width * len(models) / 2, len(models)
)

# Plotting the deviations with renamed categories and custom model colors
fig, ax = plt.subplots(figsize=(20, 9), dpi=300)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, model in enumerate(models):
    for j, category in enumerate(renamed_categories):
        # deviation = deviations[category][model]
        baseline = baseline_scores[model]
        # bottom = baseline if deviation >= 0 else baseline + deviation
        # height = abs(deviation)
        plt.bar(
            j + offsets[i],
            baseline,
            bar_width,
            bottom=0,
            color=custom_colors[i],
            label=model,
            # fontsize=17,
        )
        # if deviation < 0:
        #     plt.text(
        #         j + offsets[i],
        #         baseline + deviation - 0.16,
        #         f"{baseline + deviation:.1f}",
        #         ha="center",  # Horizontal alignment
        #         va="bottom",  # Vertical alignment
        #         fontsize=12,
        #     )
        # else:
        #     plt.text(
        #         j + offsets[i],
        #         baseline + deviation,
        #         f"{baseline + deviation:.1f}",
        #         ha="center",  # Horizontal alignment
        #         va="bottom",  # Vertical alignment
        #         fontsize=12,
        #     )


left_color = "#E0FFD6"
right_color = "#E6E6FA"
# Drawing braces and labels outside of the plot area
brace1, brace_annotation1, text1, text_annotation1 = draw_brace(
    ax,
    (-0.45, 4.45),
    "Current",
    y_offset=1.2,
    y_height=-0.08,
    color="#006400",
    draw_brace=True,
)
brace2, brace_annotation2, text2, text_annotation2 = draw_brace(
    ax,
    (3.55, 6.45),
    "Anticipated",
    y_offset=0.73,
    y_height=-0.08,
    color="#7B1FA2",
    draw_brace=True,
)
brace3, brace_annotation3, text3, text_annotation3 = draw_brace(
    ax,
    (3.55, 6.45),
    "Actual",
    y_offset=0.73,
    y_height=-0.08,
    color="#7B1FA2",
    draw_brace=True,
)

# Add vertical lines and baselines with matching colors
for x in range(len(renamed_categories) - 1):
    plt.axvline(x + 0.5, color="gray", linestyle="--", lw=1, alpha=0.2)
# for model, baseline in baseline_scores.items():
#     plt.hlines(
#         baseline,
#         xmin=-0.5,
#         xmax=len(renamed_categories) - 0.5,
#         colors=color_alpha_dict.get(model[:-4], "gray"),
#         linestyles="dotted",
#     )
#     if (
#         model == "llama-2-70b_res"
#         or model == "gpt-3.5-turbo-instruct_res"
#         or model == "gpt-4_res"
#     ):
#         plt.text(
#             # Position at the right end of the plot
#             len(renamed_categories) - 0.5,
#             baseline - 0.13,
#             f"{baseline:.2f}",  # Display the baseline value
#             va="center",  # Vertical alignment
#             ha="right",  # Horizontal alignment
#             fontsize=15,
#             color=color_alpha_dict.get(model[:-4], "gray"),
#         )
#     else:
#         plt.text(
#             # Position at the right end of the plot
#             len(renamed_categories) - 0.5,
#             baseline + 0.1,
#             f"{baseline:.2f}",  # Display the baseline value
#             va="center",  # Vertical alignment
#             ha="right",  # Horizontal alignment
#             fontsize=15,
#             color=color_alpha_dict.get(model[:-4], "gray"),
#         )
# separator_index = len(categories_group_1) - 0.5
# # plt.axvline(separator_index, color="crimson", linestyle="-", linewidth=2)


ax.axvspan(
    xmin=-0.5,
    xmax=len(categories_group_1) - 0.5,
    ymin=0,
    ymax=1,
    color=left_color,
    alpha=0.4,
)

ax.axvspan(
    xmin=len(categories_group_1) - 0.5,
    xmax=len(renamed_categories) - 0.5,
    ymin=0,
    ymax=1,
    color=right_color,
    alpha=0.4,
)

# Finalizing the plot with custom x-axis and colors
plt.xticks(
    range(len(renamed_categories)),
    renamed_categories,
    rotation=0,
    fontsize=20,
    ha="center",
)
plt.yticks(
    np.arange(3, 9, 1),
    fontsize=20,
)
plt.ylim(3, 8)
plt.tight_layout(pad=0.01)

plt.xlim(-0.5, len(renamed_categories) - 0.5)
plt.legend(
    bbox_to_anchor=(0.5, 1.17),
    loc="upper center",
    ncol=len(models) / 2 + 1,
    fontsize=17,
)
# plt.title("Influence of Other Factors on Agent Trust", fontsize=25)
plt.ylabel("Z-Scored(Mean)", fontsize=25)
# plt.xlabel("Category")
plt.tight_layout()  # Adjust layout for legend below
# plt.savefig(
# )
plt.show()
