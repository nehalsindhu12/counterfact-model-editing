import json
import numpy as np
import os
import matplotlib.pyplot as plt

most_orthogonal_result = "/data/nehals/code/editing/edit-results/EMMET/run_001/1_0_edits-case_15746.json"
least_orthogonal_result = "/data/nehals/code/editing/edit-results/EMMET/run_000/1_0_edits-case_15746.json"

with open(most_orthogonal_result, "r") as f:
    most_result = json.load(f)
with open(least_orthogonal_result, "r") as f:
    least_result = json.load(f)

def extract_metrics(result):
    """Extract relevant evaluation metrics."""
    metrics = {
        "paraphrase_score": np.mean([entry["target_new_prob"] - entry["target_true_prob"] for entry in result["post"]["paraphrase_prompts_probs"]]),
        "neighborhood_score": np.mean([entry["target_new_prob"] - entry["target_true_prob"] for entry in result["post"]["neighborhood_prompts_probs"]]),
        "efficacy_score": np.mean([entry["target_new_prob"] > entry["target_true_prob"] for entry in result["post"]["rewrite_prompts_probs"]]),
        "ngram_entropy": result["post"]["ngram_entropy"],
        "adjusted_token_entropy": result["post"]["adjusted_token_entropy"],
        "reference_score": result["post"]["reference_score"]
    }
    return metrics

most_metrics = extract_metrics(most_result)
least_metrics = extract_metrics(least_result)

metrics_file = "/data/nehals/code/editing/metrics/comparison_metrics.json"
os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

with open(metrics_file, "w") as f:
    json.dump({"most_orthogonal": most_metrics, "least_orthogonal": least_metrics}, f, indent=4)

print(f"Metrics saved to {metrics_file}")

metrics_labels = list(most_metrics.keys())
most_values = list(most_metrics.values())
least_values = list(least_metrics.values())

x = np.arange(len(metrics_labels))
width = 0.35

plt.figure(figsize=(15, 9))
plt.bar(x - width/2, most_values, width, label="Most Orthogonal", alpha=0.7)
plt.bar(x + width/2, least_values, width, label="Least Orthogonal", alpha=0.7)

plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Comparison of Editing Performance: Most vs. Least Orthogonal")
plt.xticks(x, metrics_labels, rotation=30)
plt.legend()
plt.grid(axis="y")

plot_file = "/data/nehals/code/editing/plots/orthogonal_comparison.png"
os.makedirs(os.path.dirname(plot_file), exist_ok=True)
plt.savefig(plot_file)
plt.tight_layout()
plt.show()

print(f"Plot saved to {plot_file}")
