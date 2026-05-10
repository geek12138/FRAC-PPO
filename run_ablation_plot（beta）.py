import os
import subprocess
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

# === 实验配置 ===
beta_list = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

# 其他固定参数
fixed_zeta = 0.1
runs = 1
epochs = 1000
L_num = 200
python_script = "main_PPO_stat_box_save.py"

# 存储 JSON 文件路径的列表
data_files = []

print("=== 开始运行 Beta 消融实验 ===")

for beta in beta_list:
    print(f"\n>>> 正在运行 beta = {beta} ...")

    current_seed = np.random.randint(1, 100000)

    cmd = [
        "python", python_script,
        "-zeta", str(fixed_zeta),
        "-beta", str(beta),
        "-runs", str(runs),
        "-epochs", str(epochs),
        "-L_num", str(L_num),
        "-seed", str(current_seed),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running beta={beta}: {e}")
        continue

    search_pattern = f"data/PPO_stat_box_*_zeta_{fixed_zeta}_beta_{beta}"
    dirs = sorted(glob.glob(search_pattern), key=os.path.getmtime)

    if dirs:
        latest_dir = dirs[-1]
        json_file = os.path.join(latest_dir, "final_fractions_data.json")
        if os.path.exists(json_file):
            data_files.append((beta, json_file))
            print(f" [Success] 数据已记录: {json_file}")
        else:
            print(f" [Warning] 未找到数据文件: {json_file}")
    else:
        print(f" [Error] 未找到输出目录: {search_pattern}")

print("\n=== 所有实验运行结束，开始绘图 ===")

plt.figure(figsize=(10, 8))

markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(data_files), 1)))

for i, (beta, file_path) in enumerate(data_files):
    with open(file_path, 'r') as f:
        data = json.load(f)

    r_values = data['r_values']
    c_means = data['C_means']

    plt.plot(
        r_values, c_means,
        marker=markers[i % len(markers)],
        color=colors[i],
        markersize=8,
        markerfacecolor='none',
        markeredgewidth=1.5,
        linewidth=2,
        label=f'$\\beta = {beta}$'
    )

plt.xlabel('r', fontsize=14)
plt.ylabel('Fractions', fontsize=14)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='lower right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

out_png = 'Ablation_Beta_Result.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"绘图完成！图片已保存为: {out_png}")