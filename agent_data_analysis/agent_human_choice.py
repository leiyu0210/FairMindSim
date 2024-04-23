def plot_average_amount_sent_with_grid_below_and_legend_on_top(
    data_dictator, data_trust, color_dict, save_path=None, front_size=17
):
    """
    Plots a horizontal bar chart with grid lines behind the bars and the legend on the top.
    Uses different patterns for dictator game and trust game, with specified colors for each model.

    :param data_dictator: A dictionary containing the amounts sent for each model in the dictator game.
    :param data_trust: A dictionary containing the amounts sent for each model in the trust game.
    :param color_dict: A dictionary specifying colors for each model.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    # 计算平均值
    avg_dictator = {model: np.mean(values) for model, values in data_dictator.items()}
    avg_trust = {model: np.mean(values) for model, values in data_trust.items()}

    # 模型名称
    models = list(avg_dictator.keys())
    index = models.index("llama-2-7b")
    models.pop(index)
    models.sort(key=lambda x: avg_trust[x] - avg_dictator[x])

    # 创建柱状图
    bar_width = 0.4
    index = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, model in enumerate(models):
        if "llama-7b" in model.lower():
            continue
        # 模型颜色
        color = color_dict.get(model, "gray")  # 如果没有指定颜色，则使用灰色

        ax.bar(
            index[idx], avg_dictator[model], bar_width, color=color, zorder=2, alpha=0.5
        )
        ax.bar(
            index[idx],
            avg_dictator[model],
            bar_width,
            color="none",
            hatch="x",
            edgecolor="black",
            zorder=3,
        )
        ax.bar(
            index[idx] + bar_width,
            avg_trust[model],
            bar_width,
            color=color,
            zorder=3,
            alpha=0.5,
        )
        ax.bar(
            index[idx] + bar_width,
            avg_trust[model],
            bar_width,
            color="none",
            zorder=3,
            edgecolor="black",
        )

        # 在每个柱子上添加数字
        ax.text(
            index[idx],
            avg_dictator[model],
            f"{avg_dictator[model]:.1f}",
            ha="center",
            va="bottom",
            fontsize=front_size - 2,
            zorder=3,
        )
        ax.text(
            index[idx] + bar_width,
            avg_trust[model],
            f"{avg_trust[model]:.1f}",
            ha="center",
            va="bottom",
            fontsize=front_size - 2,
            zorder=3,
        )

    # 添加自定义图例
    legend_elements = [
        mpatches.Patch(
            facecolor="none", hatch="x", edgecolor="black", label="Dictator Game"
        ),
        mpatches.Patch(facecolor="none", edgecolor="black", label="Trust Game"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=front_size,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
    )

    # 添加虚线网格并设置为底层
    ax.yaxis.grid(
        True, linestyle="--", which="major", color="grey", alpha=0.4, zorder=-1
    )
    ax.set_axisbelow(True)
    plt.rcParams["axes.axisbelow"] = True
    # 设置图表标签
    ax.set_ylabel("Average Amount Sent ($)", fontsize=front_size + 5)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(rename_model_name, fontsize=front_size)
    ax.set_yticklabels(range(0, 8), fontsize=front_size)

    # 移除图表的顶部和右侧边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 显示图表
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
