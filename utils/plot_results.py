import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

task_data = defaultdict(list)
summary_data = defaultdict(list)

# base_dir = "/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/cfg/08-18_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Eval_Model_ckpt_100000/libero_10"
base_dir = "/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/exp/08-26_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Eval_Model_ckpt_100000/libero_10"
result_dir = os.path.join(base_dir, "results.json")
use_parsing = False
parsing_key = "// w_cfg = 2.0"
with open(result_dir, 'r') as f:
    start_parsing = False
    count = 0
    for line in f:
        line = line.strip()
        if not line:
            continue
        if parsing_key in line:
            start_parsing = True
            continue  # Move to the next line after finding the start comment
        if start_parsing and line.startswith("//"):
            break
        if start_parsing or not use_parsing:
            try:
                data = json.loads(line)
                for key, value in data.items():
                    if key.startswith("summary"):
                        summary_data[key].append(value)
                    elif key.startswith("libero"):
                        task_data[key].append(value)
            except json.JSONDecodeError:
                continue
            count += 1
# print(count)
task_stats = {}
sorted_task_keys = task_data.keys()

# Create task labels
task_labels = {key: f"task{i + 1}" for i, key in enumerate(sorted_task_keys)}

for key in sorted_task_keys:
    scores = np.array(task_data[key])
    mean = np.mean(scores)
    var = np.var(scores)
    task_stats[key] = {'mean': mean, 'var': var}

# Calculate overall success rate statistics
if summary_data:
    summary_key = list(summary_data.keys())[0]
    summary_scores = np.array(summary_data[summary_key])
    summary_mean = np.mean(summary_scores)
    summary_var = np.var(summary_scores)
else:
    summary_mean = 0.0
    summary_var = 0.0

print("\n**Overall Success Rate:**")
print(f"Mean: {summary_mean:.2f}")
print(f"Var: {summary_var:.4f}")
print("\n**Per-Task Success Rate:**")
print("\n| Task  | Mean |  Var   |")
table_data = task_stats.items()
for key, stats in table_data:
    print(f"| {task_labels[key]} | {stats['mean']:.2f} | {stats['var']:.4f} |")

if task_stats:
    # Prepare data for plotting, sorted by mean success rate
    plot_data = []
    for key in sorted_task_keys:
        stats = task_stats[key]
        std_dev = np.sqrt(stats['var'])
        plot_data.append((task_labels[key], stats['mean'], std_dev))

    sorted_labels = [item[0] for item in plot_data]
    sorted_means = [item[1] for item in plot_data]
    sorted_std_devs = [item[2] for item in plot_data]

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(sorted_labels, sorted_means, yerr=sorted_std_devs,
                  capsize=5, color='skyblue', edgecolor='black')

    ax.set_title('Average Success Rate by Task', fontsize=18, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Average Success Rate', fontsize=12)
    ax.set_ylim(0, 1.5)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("task_success_rates.png")
    print("Saved a graph to task_success_rates.png")
else:
    print("\n---")
    print("No task data found to generate graph")