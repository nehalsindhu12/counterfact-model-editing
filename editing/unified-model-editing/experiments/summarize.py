import collections
import json
from pprint import pprint
from typing import List, Optional
import sys

import numpy as np
from scipy.stats import hmean

sys.path.append('/home/akshatgupta/KnowledgeEditing_local/unified-model-editing')
from util.globals import *


def summarize(
    dir_name = None,
    runs: Optional[List] = None,
    first_n_cases=None,
    abs_path=False,
    get_uncompressed=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
    #for run_dir in [Path(abs_path)]:

        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        file_wise_results = {}
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            #if first_n_cases is not None and case_id >= first_n_cases:
            #    break

            num_edits = int(str(case_file).split('/')[-1].split('_')[1])
            if first_n_cases is not None and num_edits >= first_n_cases:
                continue

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["post_target_true_prob"] > x["post_target_new_prob"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["post_target_new_prob"]) - np.exp(-x["post_target_true_prob"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["post_target_true_prob"] < x["post_target_new_prob"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["post_target_true_prob"]) - np.exp(-x["post_target_new_prob"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Accuracy-based evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                # (
                #     f"{prefix}_rewrite_acc",
                #     f"{prefix}_paraphrase_acc",
                #     f"{prefix}_neighborhood_acc",
                # ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],
                        cur_sum[k_generalization][0],
                        cur_sum[k_specificity][0],
                    ]

                    # if f"{prefix}_ngram_entropy" in cur_sum:
                    #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                    # if f"{prefix}_reference_score" in cur_sum:
                    #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])

                    cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                    break

        print(metadata)
        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    #return uncompressed if get_uncompressed else summaries
    return cur_sum


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    summarize(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
        args.path
    )
