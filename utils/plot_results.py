import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

task_data = defaultdict(list)
summary_data = defaultdict(list)

# base_dir = "/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/cfg/08-18_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Eval_Model_ckpt_100000/libero_10"
base_dir = "/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline/08-11_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Eval_Model_ckpt_100000/libero_10"
result_dir = os.path.join(base_dir, "results.json")
use_parsing = False
parsing_key = "// w_cfg = 2.0"
start_parsing = False
count = 0
with open(result_dir, 'r') as f:
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
                    if key.startswith("sim_summary"):
                        summary_data[key].append(value)
                    elif key.startswith("sim/"):
                        task_data[key].append(value)
            except json.JSONDecodeError:
                continue
print(count)
# 3. Calculate statistics
task_stats = {}
# Sort keys to ensure consistent task order
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

# 4. Print results
print("### Statistical Results ###")
print("\n**Overall Success Rate:**")
print(f"Mean: {summary_mean:.2f}")
print(f"Var: {summary_var:.4f}")
print("**Per-Task Success Rate:**")
print("\n| Task  | Mean |  Var   |")
# Sort tasks by mean success rate for the table
table_data = task_stats.items()
# table_data = sorted(task_stats.items(), key=lambda item: item[1]['mean'], reverse=True)
for key, stats in table_data:
    print(f"| {task_labels[key]} | {stats['mean']:.2f} | {stats['var']:.4f} |")

# 5. Generate graph
if task_stats:
    # Prepare data for plotting, sorted by mean success rate
    plot_data = []
    for key in sorted_task_keys:
        stats = task_stats[key]
        std_dev = np.sqrt(stats['var'])
        plot_data.append((task_labels[key], stats['mean'], std_dev))

    # Sort by mean value in descending order
    # plot_data.sort(key=lambda x: x[1], reverse=True)

    # Unpack sorted data
    sorted_labels = [item[0] for item in plot_data]
    sorted_means = [item[1] for item in plot_data]
    sorted_std_devs = [item[2] for item in plot_data]

    # Create the plot
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(sorted_labels, sorted_means, yerr=sorted_std_devs,
                  capsize=5, color='skyblue', edgecolor='black')

    ax.set_title('Average Success Rate by Task', fontsize=18, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Average Success Rate', fontsize=12)
    ax.set_ylim(0, 1.5)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add mean value text on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f'{yval:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("task_success_rates.png")

    print("Generated Graph")
else:
    print("\n---")
    print("No task data found to generate graph")