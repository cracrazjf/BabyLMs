import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_analysis_1():
    df = pd.read_csv("./outputs/analysis1_cell_means__20260315_093451.csv")

    # Rename for readability
    df["measure"] = df["measure"].map({
        "sum logprob": "Logprob",
        "embed sim": "Embedding"
    })

    df["relationship"] = df["relationship"].str.capitalize()
    df["condition"] = df["condition"].str.capitalize()

    relationships = ["Cohyponym", "Superordinate"]

    # Scientific color palette (Okabe–Ito)
    colors = {
        "Easy": "#0072B2",
        "Hard": "#D55E00"
    }

    plt.rcParams.update({
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, axes = plt.subplots(1, 2, figsize=(9,4), sharey=True)
    x_positions = [0, 0.4]

    for ax, measure in zip(axes, ["Logprob", "Embedding"]):

        subset = df[df["measure"] == measure]

        for cond in ["Easy","Hard"]:
            cond_data = subset[subset["condition"] == cond]
            cond_data = cond_data.set_index("relationship").loc[relationships]

            x = [0, 0.4]

            y = cond_data["p_mean"]

            yerr = [
                y - cond_data["ci_lo"],
                cond_data["ci_hi"] - y
            ]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=7,
                linewidth=2,
                color=colors[cond],
                capsize=4,
                label=cond
            )

        ax.set_title(measure)
        ax.set_xticks(range(len(relationships)))
        ax.set_xticklabels(relationships)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(0.5, 1.01)
        ax.set_xticks(x)
        ax.set_xticklabels(relationships)
        ax.set_xlim(-0.15, 0.55)

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(title="Difficulty", frameon=False)

    plt.tight_layout()
    plt.savefig("./outputs/accuracy_plot.png", dpi=300)
    plt.close()

def plot_analysis_2():
    df = pd.read_csv("./outputs/analysis2_cell_means__20260315_005000.csv")

    # Clean labels
    df["relationship"] = df["relationship"].str.capitalize()
    df["condition"] = df["condition"].str.capitalize()

    df["prompt_type"] = df["prompt_type"].replace({
        "control": "Control",
        "task specific": "Task-specific"
    })

    df["meta_prompt_type"] = df["meta_prompt_type"].replace({
        "none": "None",
        "neutral": "Neutral",
        "task specific": "Task-specific"
    })

    meta_order = ["None", "Neutral", "Task-specific"]
    relationships = ["Cohyponym", "Superordinate"]

    # Bar colors (scientific palette)
    colors = {
        ("Control","Easy"): "#bdbdbd",
        ("Control","Hard"): "#737373",
        ("Task-specific","Easy"): "#9ecae1",
        ("Task-specific","Hard"): "#2171b5"
    }

    combos = [
        ("Control","Easy"),
        ("Control","Hard"),
        ("Task-specific","Easy"),
        ("Task-specific","Hard")
    ]

    plt.rcParams.update({
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, axes = plt.subplots(1,2, figsize=(11,5), sharey=True)

    bar_width = 0.18
    x = np.arange(len(meta_order))

    for ax, rel in zip(axes, relationships):

        sub = df[df["relationship"] == rel]

        for i,(prompt,cond) in enumerate(combos):

            d = sub[
                (sub["prompt_type"] == prompt) &
                (sub["condition"] == cond)
            ].copy()

            d["meta_prompt_type"] = pd.Categorical(
                d["meta_prompt_type"],
                categories=meta_order,
                ordered=True
            )

            d = d.sort_values("meta_prompt_type")

            y = d["p_mean"].values
            yerr = [
                y - d["ci_lo"].values,
                d["ci_hi"].values - y
            ]

            offset = (i - 1.5) * bar_width

            ax.bar(
                x + offset,
                y,
                width=bar_width,
                color=colors[(prompt,cond)],
                edgecolor="black",
                linewidth=0.5,
                label=f"{prompt}, {cond}"
            )

            ax.errorbar(
                x + offset,
                y,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1
            )

        ax.set_title(rel)
        ax.set_xticks(x)
        ax.set_xticklabels(meta_order)
        ax.set_xlabel("Metaprompt type")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(0.5,1.01)

    axes[0].set_ylabel("Accuracy")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False
    )

    plt.subplots_adjust(top=0.82)
    plt.savefig("./outputs/analysis2_plot.png", dpi=300)
    plt.close()

def plot_analysis_3():

    df = pd.read_csv("./outputs/analysis3_cell_means__20260315_093451.csv")

    df["relationship"] = df["relationship"].str.capitalize()
    df["condition"] = df["condition"].str.capitalize()

    name_map = {
        "gpt2": "GPT-2",
        "mistral_7b_base": "Mistral-B",
        "mistral_7b_instruct": "Mistral-I",
        "gemma2_9b_base": "Gemma",
        "llama3.1_8b_base": "LLaMA-B",
        "llama3.1_8b_instruct": "LLaMA-I",
        "qwen3_8b_base": "Qwen-B",
        "qwen3_8b_instruct": "Qwen-I",
        # "olmo3_7b_base": "OLMo-B",
        # "olmo3_7b_instruct": "OLMo-I"
    }
    df["model"] = df["model"].map(name_map)

    base_models = ["GPT-2", "Mistral-B", "Gemma", "LLaMA-B", "Qwen-B"]
    instruct_models = ["Mistral-I", "LLaMA-I", "Qwen-I"]
    model_order = base_models + instruct_models

    # Order of the 4 bars within each model
    combo_order = [
        ("Cohyponym", "Easy"),
        ("Cohyponym", "Hard"),
        ("Superordinate", "Easy"),
        ("Superordinate", "Hard"),
    ]

    # Colorblind-safe palette
    combo_colors = {
        ("Cohyponym", "Easy"): "#56B4E9",
        ("Cohyponym", "Hard"): "#0072B2",
        ("Superordinate", "Easy"): "#E69F00",
        ("Superordinate", "Hard"): "#D55E00",
    }

    plt.rcParams.update({
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)

    bar_width = 0.18
    group_gap = 0.35

    for ax, measure, title in zip(
        axes,
        ["sum logprob", "embed sim"],
        ["Logprob", "Embedding"]
    ):
        sub = df[df["measure"] == measure].copy()

        group_centers = []
        x = 0.0

        for model in model_order:
            group_centers.append(x)

            for j, (rel, cond) in enumerate(combo_order):
                row = sub[
                    (sub["model"] == model) &
                    (sub["relationship"] == rel) &
                    (sub["condition"] == cond)
                ].iloc[0]

                xpos = x + (j - 1.5) * bar_width
                y = row["p_mean"]
                yerr = [[y - row["ci_lo"]], [row["ci_hi"] - y]]

                ax.bar(
                    xpos,
                    y,
                    width=bar_width,
                    color=combo_colors[(rel, cond)],
                    edgecolor="none"
                )
                ax.errorbar(
                    xpos,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=2
                )

            x += 4 * bar_width + group_gap

        ax.set_xticks(group_centers)
        ax.set_xticklabels(model_order, rotation=40, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.5, 1.01)

    # Vertical divider between base and instruct
    divider_x = (group_centers[len(base_models)-1] + group_centers[len(base_models)]) / 2
    for ax in axes:
        ax.axvline(divider_x, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # Group labels
    for ax in axes:
        ax.text(
            np.mean(group_centers[:len(base_models)]), 1.005, "Base models",
            ha="center", va="bottom", fontsize=10
        )
        ax.text(
            np.mean(group_centers[len(base_models):]), 1.005, "Instruction-tuned",
            ha="center", va="bottom", fontsize=10
        )

    handles = [
    plt.Rectangle((0, 0), 1, 1, color=combo_colors[k])
    for k in combo_order
    ]

    labels = [
        "Cohyponym – Easy",
        "Cohyponym – Hard",
        "Superordinate – Easy",
        "Superordinate – Hard"
    ]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("./outputs/analysis3_plot.png", dpi=300)
    plt.close()

def main():
    plot_analysis_1()
    plot_analysis_2()   
    plot_analysis_3()

if __name__ == "__main__":
    main()