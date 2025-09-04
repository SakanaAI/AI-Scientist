import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire


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


def main(path_to_arc_tasks: str = "./experiments/arc2/arc_agi_2_eval_short.txt"):
    top_k = 2
    save_path: str = "./outputs/arc2/{exp_name}/"

    if not Path("./outputs/plots").exists():
        Path("./outputs/plots").mkdir()

    with open(path_to_arc_tasks, "r") as f:
        task_list = f.readlines()
    task_list = [t.strip() for t in task_list]
    N_TASKS = len(task_list)

    exp_name_list = os.environ["exp_names_str"].split(",")
    for ret_type in ["pass@2", "coverage"]:
        save_path_i = f"./outputs/plots/{ret_type}"
        if "pass@2" in ret_type:
            save_path_i += f"_top{top_k}"
        save_path_i += ".png"
        plt.figure(figsize=(8, 7))
        plt.grid(True)
        for exp_name in exp_name_list:
            if "coverage" in ret_type:
                df_test = pd.read_csv(
                    os.path.join(save_path.format(exp_name=exp_name), "df_test.csv")
                )
                df_score_i = get_coverage(df_test)
            else:
                df_test = pd.read_csv(
                    os.path.join(save_path.format(exp_name=exp_name), "df_test.csv")
                )
                df_reward = pd.read_csv(
                    os.path.join(save_path.format(exp_name=exp_name), "df_reward.csv")
                )
                df_score_i = get_test_score_by_reward_topk(
                    df_test, df_reward, is_early_prioritize=False, top_k=top_k
                )
            x = range(len(df_score_i))
            y = (df_score_i == 1).sum(axis=1).values / N_TASKS

            plt.plot(
                x,
                y,
                label=f"{exp_name} ({df_score_i.shape[1]}), max score: {100 * y.max():.1f}%",
            )

        plt.title(f"ARC-AGI-2, AB-MCTS")
        plt.xlabel("Generation Budget")
        if ret_type == "coverage":
            plt.ylabel(f"Pass@k (Coverage)")
        else:
            plt.ylabel(f"Pass@2")
        plt.legend()
        plt.savefig(save_path_i)
        plt.close()


if __name__ == "__main__":
    Fire(main)
