import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("./experiments/arc2")
warnings.filterwarnings("ignore")


def get_coverage(df, is_lower_better=False):
    array_df = df.copy().values
    for i in range(len(array_df) - 1):
        if is_lower_better:
            array_df[i + 1] = np.minimum(array_df[i], array_df[i + 1])
        else:
            array_df[i + 1] = np.maximum(array_df[i], array_df[i + 1])
    return pd.DataFrame(array_df, index=df.index, columns=df.columns)


def get_perfect_coverage(df_test, df_reward, is_lower_better=False):
    array_df = df_test.copy().values * df_reward.copy().values
    for i in range(len(array_df) - 1):
        if is_lower_better:
            array_df[i + 1] = np.minimum(array_df[i], array_df[i + 1])
        else:
            array_df[i + 1] = np.maximum(array_df[i], array_df[i + 1])
    return pd.DataFrame(array_df, index=df_test.index, columns=df_test.columns)


def get_test_score_by_reward_topk(
    df_test,
    df_reward,
    top_k: int = 1,
    is_early_prioritize: bool = False,
    is_lower_better: bool = False,
):
    """
    Selects the best test scores from the top_k candidates with highest reward (val),
    returning a time-series DataFrame (≈ pass@k).
    """
    arr_test = df_test.values.copy()
    arr_reward = df_reward.values.copy()
    arr_result = arr_test.copy()

    n_rows, n_cols = arr_test.shape
    k = max(1, top_k)

    for i in range(1, n_rows):
        window_reward = arr_reward[: i + 1]
        idx = np.repeat(np.arange(i + 1)[:, None], n_cols, axis=1)

        if is_early_prioritize:
            if is_lower_better:
                sort_idx = np.lexsort((idx, window_reward), axis=0)
            else:
                sort_idx = np.lexsort((idx, -window_reward), axis=0)
        else:
            if is_lower_better:
                sort_idx = np.lexsort((-idx, window_reward), axis=0)
            else:
                sort_idx = np.lexsort((-idx, -window_reward), axis=0)

        topk_idx = sort_idx[:k].T

        row_out = []
        for test_col, idx_k in zip(arr_test[: i + 1].T, topk_idx, strict=True):
            sel_scores = test_col[idx_k]
            best = np.nanmin(sel_scores) if is_lower_better else np.nanmax(sel_scores)
            row_out.append(best)

        arr_result[i] = row_out

    return pd.DataFrame(arr_result, index=df_test.index, columns=df_test.columns)


def main():
    folders = os.listdir("./")
    final_results = {}
    for folder in folders:
        if folder.startswith("run") and os.path.isdir(folder):
            with open(os.path.join(folder, "final_info.json")) as f:
                final_results[folder] = json.load(f)

    # CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
    labels = {
        "run_0": "Baseline",
        "run_1": "Code Diversity (α=0.3)",
        "run_2": "Code Diversity (α=0.5)",
        "run_3": "Output Diversity (α=0.3)",
        "run_4": "Output Diversity (α=0.5)",
    }

    top_k = 2
    save_path = "./plots"

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    for ret_type in ["pass@2", "coverage", "perfect_coverage"]:
        save_path_i = f"./plots/{ret_type}"
        if "pass@2" in ret_type:
            save_path_i += f"_top{top_k}"
        save_path_i += ".png"
        plt.figure(figsize=(8, 7))
        plt.grid(True)

        for folder, final_result in final_results.items():
            result_dir = Path(folder)
            n_tasks = final_result["arc_agi"]["num_tasks"]

            if ret_type == "coverage":
                df_test = pd.read_csv(os.path.join(result_dir, "df_test.csv"))
                df_score_i = get_coverage(df_test)
            elif ret_type == "perfect_coverage":
                df_test = pd.read_csv(os.path.join(result_dir, "df_test.csv"))
                df_reward = pd.read_csv(os.path.join(result_dir, "df_reward.csv"))
                df_score_i = get_perfect_coverage(df_test, df_reward)
            else:  # pass@2
                df_test = pd.read_csv(os.path.join(result_dir, "df_test.csv"))
                df_reward = pd.read_csv(os.path.join(result_dir, "df_reward.csv"))
                df_score_i = get_test_score_by_reward_topk(df_test, df_reward, is_early_prioritize=False, top_k=top_k)
            x = range(len(df_score_i))
            y = (df_score_i == 1).sum(axis=1).values / n_tasks

            plt.plot(
                x,
                y,
                label=f"{labels[folder]}, max score: {100 * y.max():.1f}%",
            )

        plt.title("Effect of Diversity Bonus on ARC-AGI Task Solving")
        plt.xlabel("Generation Budget")
        if ret_type == "coverage":
            plt.ylabel("Test Accuracy (Coverage)")
        elif ret_type == "perfect_coverage":
            plt.ylabel("Perfect Accuracy (Coverage)")
        else:
            plt.ylabel("Pass@2")
        plt.legend()
        plt.savefig(save_path_i)
        plt.close()


if __name__ == "__main__":
    main()
