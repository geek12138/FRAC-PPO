import torch
from FRAC_PPO import SPGG
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import argparse
import os
import shutil
import re
from scipy import stats
import pandas as pd
import asyncio


def calculate_statistics(data):
    """计算均值、标准差和95%置信区间"""
    mean = np.mean(data)
    std = np.std(data, ddof=1) if len(data) > 1 else 0.0
    n = len(data)
    if n > 1:
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
    else:
        ci = (mean, mean)
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }


def plot_cooperation_boxplot(stats_results, output_path, fontsize=24):
    """
    绘制最终合作率的箱线图
    :param stats_results: 包含所有r值的统计结果字典（需含'raw_data'字段）
    :param output_path: 输出目录路径
    :param fontsize: 字体大小
    """
    r_values = sorted(stats_results.keys())

    min_r = min(r_values)
    max_r = max(r_values)
    target_ticks = [round(min_r + i * 0.5, 1) for i in range(int((max_r - min_r) / 0.5) + 1)]
    existing_ticks = [r for r in target_ticks if r in r_values]

    cooperation_data = [stats_results[r]['C']['raw_data'] for r in r_values]

    plt.figure(figsize=(12, 6))

    box = plt.boxplot(
        cooperation_data,
        positions=np.arange(len(r_values)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True
    )

    colors = ['lightblue'] * len(r_values)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.setp(box['medians'], color='darkblue', linewidth=2)
    plt.setp(box['means'], color='red', linestyle='--', linewidth=2)

    tick_positions = [r_values.index(r) for r in existing_ticks]
    plt.xticks(
        tick_positions,
        [f'{r:.1f}' for r in existing_ticks],
        fontsize=fontsize
    )

    plt.yticks(fontsize=fontsize)
    plt.xlabel('r', fontsize=fontsize)
    plt.ylabel('Final Cooperation Fractions', fontsize=fontsize)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkblue', lw=2, label='Median'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, label='Mean'),
        Line2D([0], [0], marker='o', color='black', markerfacecolor='w',
               markersize=6, linestyle='None', label='Outlier')
    ]

    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=fontsize
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(
        f'{output_path}/final_cooperation_boxplot.png',
        dpi=300, bbox_inches='tight', pad_inches=0
    )

    plt.close()


def plot_cooperation_violinplot(stats_results, output_path, fontsize=24):
    """
    绘制最终合作率的小提琴图
    :param stats_results: 包含所有r值的统计结果字典（需含'raw_data'字段）
    :param output_path: 输出目录路径
    :param fontsize: 字体大小
    """
    r_values = sorted(stats_results.keys())

    min_r = min(r_values)
    max_r = max(r_values)
    target_ticks = [round(min_r + i * 0.5, 1) for i in range(int((max_r - min_r) / 0.5) + 1)]
    existing_ticks = [r for r in target_ticks if r in r_values]

    cooperation_data = [stats_results[r]['C']['raw_data'] for r in r_values]

    plt.figure(figsize=(12, 6))

    parts = plt.violinplot(
        cooperation_data,
        positions=np.arange(len(r_values)),
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')

    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('darkblue')

    tick_positions = [r_values.index(r) for r in existing_ticks]
    plt.xticks(
        tick_positions,
        [f'{r:.1f}' for r in existing_ticks],
        fontsize=fontsize
    )

    plt.yticks(fontsize=fontsize)
    plt.xlabel('r', fontsize=fontsize)
    plt.ylabel('Final Cooperation Fractions', fontsize=fontsize)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkblue', lw=2, label='Median'),
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=8, label='Distribution')
    ]

    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=fontsize
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(
        f'{output_path}/final_cooperation_violinplot.png',
        dpi=300, bbox_inches='tight', pad_inches=0
    )

    plt.close()


def save_params_to_json(params, filename_prefix="params", output_path='data'):
    param_dir = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = param_dir / filename

    serializable_params = {
        k: str(v) if isinstance(v, torch.device) else v
        for k, v in params.items()
    }

    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)

    src_file = 'main_PPO_Quota.py'
    dst_file = f'{output_path}/{src_file}'
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)

    src_file = 'PPO_Quota.py'
    dst_file = f'{output_path}/{src_file}'
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)

    print(f"参数已保存至: {filepath}")


async def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M%S")
    fontsize = 16

    r_values = [round(i * 0.1, 1) for i in range(25, 56)]

    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'mps':
        device = torch.device("mps")
    else:
        raise ValueError(f"Unsupported device: {args.device}")

    if args.epochs == 1000:
        xticks = [0, 1, 10, 100, 1000]
    elif args.epochs == 2000:
        xticks = [0, 1, 10, 100, 1000, 2000]
    elif args.epochs == 10000:
        xticks = [0, 1, 10, 100, 1000, 10000]
    elif args.epochs == 100000:
        xticks = [0, 1, 10, 100, 1000, 10000, 100000]
    else:
        xticks = [0, 1]

    fra_yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    profite_yticks = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    experiment_params = {
        "r": r_values,
        "epochs": args.epochs,
        "runs": args.runs,
        "L_num": args.L_num,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "clip_epsilon": args.clip_epsilon,
        "question": args.question,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "gae_lambda": args.gae_lambda,
        "device": device,
        "xticks": xticks,
        "fra_yticks": fra_yticks,
        "profite_yticks": profite_yticks,
        "start_time": formatted_time,
        "seed": args.seed,
        "delta": args.delta,
        "rho": args.rho,
        "zeta": args.zeta,
        "beta": args.beta
    }

    output_path = (
        f'data/PPO_stat_box_{formatted_time}'
        f'_q{args.question}'
        f'_e_{args.epochs}'
        f'_L_{args.L_num}'
        f'_a_{args.alpha}'
        f'_g_{args.gamma}'
        f'_ce_{args.clip_epsilon}'
        f'_gl_{args.gae_lambda}'
        f'_p_{args.ppo_epochs}'
        f'_b_{args.batch_size}'
        f'_delta_{args.delta}'
        f'_rho_{args.rho}'
        f'_seed_{args.seed}'
        f'_zeta_{args.zeta}'
        f'_beta_{args.beta}'
    )

    raw_data_dir = os.path.join(output_path, 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)

    save_params_to_json(experiment_params, filename_prefix="params", output_path=output_path)

    results = {
        'strategy_evolution': {},
        'final_cooperation': {},
        'raw_cooperation': {},
        'run_data': {}
    }

    for r in r_values:
        print(f"\nRunning experiment with r={r}")
        results['strategy_evolution'][r] = {'C': [], 'D': []}
        results['final_cooperation'][r] = {'C': [], 'D': []}

        for num in range(args.runs):
            model = SPGG(
                L_num=args.L_num,
                device=device,
                alpha=args.alpha,
                gamma=args.gamma,
                clip_epsilon=args.clip_epsilon,
                r=r,
                epochs=args.epochs,
                now_time=formatted_time,
                question=args.question,
                ppo_epochs=args.ppo_epochs,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                output_path=output_path,
                delta=args.delta,
                rho=args.rho,
                zeta=args.zeta,
                beta=args.beta
            )

            print(f"Run {num + 1}/{args.runs}")
            model.count = num

            D_Y, C_Y, D_Value, C_Value, all_value = await model.run()

            results['strategy_evolution'][r]['C'].append(C_Y)
            results['strategy_evolution'][r]['D'].append(D_Y)
            results['final_cooperation'][r]['C'].append(C_Y[-1])
            results['final_cooperation'][r]['D'].append(D_Y[-1])

            if r not in results['raw_cooperation']:
                results['raw_cooperation'][r] = []
            results['raw_cooperation'][r].append(C_Y[-1])

            results['run_data'].setdefault(r, {})
            results['run_data'][r][num] = {
                'timesteps': list(range(len(C_Y))),
                'C_Y': C_Y,
                'D_Y': D_Y,
                'D_Value': D_Value,
                'C_Value': C_Value,
                'all_value': all_value
            }

            run_data_path = os.path.join(raw_data_dir, f'run_r{r}_num{num}.json')
            with open(run_data_path, 'w') as f_run:
                json.dump(results['run_data'][r][num], f_run, indent=4)

            model.save_data('Density_D', f'r{r}', r, D_Y)
            model.save_data('Density_C', f'r{r}', r, C_Y)
            model.save_data('Value_D', f'r{r}', r, D_Value)
            model.save_data('Value_C', f'r{r}', r, C_Value)
            model.save_data('Total_Value', f'r{r}', r, all_value)

            plt.clf()
            plt.close("all")
            plt.yscale('linear')
            plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
            plt.figure(figsize=(8, 6))

            plt.plot(C_Y, 'b-', linewidth=2, alpha=0.7, label='C')
            plt.plot(D_Y, 'r-', linewidth=2, alpha=0.7, label='D')

            plt.xlim(0, None)
            plt.xscale(
                'symlog',
                linthresh=1,
                linscale=0.5,
                subs=np.arange(1, 10)
            )

            plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)
            plt.yticks(fra_yticks, fontsize=fontsize)

            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.xlabel('t', fontsize=fontsize, labelpad=10)
            plt.ylabel('Fractions', fontsize=fontsize, labelpad=10)
            plt.legend(loc='best', fontsize=fontsize)

            plt.savefig(
                f'{output_path}/strategy_evolution_r{r}_run{num}.pdf',
                format='pdf', dpi=300, bbox_inches='tight', pad_inches=0
            )

            plt.close()

    print("All experiments completed!")

    with open(f'{output_path}/all_results_data.json', 'w') as f:
        json.dump({
            'strategy_evolution': results['strategy_evolution'],
            'final_cooperation': results['final_cooperation'],
            'raw_cooperation': results['raw_cooperation'],
            'run_data': results['run_data']
        }, f, indent=4)

    folder_path = f'{output_path}/Density_C'
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")] if os.path.exists(folder_path) else []

    x_values = []
    y_values = []

    for file_name in txt_files:
        match = re.search(r"r(\d+\.\d+)", file_name)
        if match:
            x_value = float(match.group(1))
            x_values.append(x_value)

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()
                try:
                    y_value = float(last_line)
                    y_values.append(y_value)
                except ValueError:
                    print(f"文件 {file_name} 的最后一行不是有效的数字: {last_line}")

    sorted_data = sorted(zip(x_values, y_values), key=lambda x: x[0])
    x_values = [x[0] for x in sorted_data]
    y_values = [x[1] for x in sorted_data]
    y_values = np.array(y_values)

    if len(x_values) > 0:
        plt.clf()
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.plot(
            x_values, y_values,
            marker="o", markersize=10, markerfacecolor='none',
            linestyle="-", color="b", label='C'
        )

        plt.plot(
            x_values, 1 - y_values,
            marker="d", markersize=10, markerfacecolor='none',
            linestyle="-", color="r", label='D'
        )

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.xlabel("r", fontsize=16)
        plt.ylabel("Fractions", fontsize=16)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(
            f'{output_path}/C_D_r.png',
            dpi=300, bbox_inches='tight', pad_inches=0
        )

        plt.close()

        cd_ratio_data = {
            'r_values': list(x_values),
            'C_values': list(y_values),
            'D_values': list(1 - y_values)
        }

        with open(f'{output_path}/cd_ratio_data.json', 'w') as f:
            json.dump(cd_ratio_data, f, indent=4)
    else:
        print("没有找到有效的数据。")

    stats_results = {}
    for r in r_values:
        stats_results[r] = {
            'C': {
                **calculate_statistics(results['final_cooperation'][r]['C']),
                'raw_data': results['raw_cooperation'][r]
            },
            'D': calculate_statistics(results['final_cooperation'][r]['D'])
        }

    boxplot_data = {}
    for r in r_values:
        boxplot_data[r] = list(stats_results[r]['C']['raw_data'])
    with open(f'{output_path}/boxplot_data.json', 'w') as f:
        json.dump(boxplot_data, f, indent=4)

    with open(f'{output_path}/statistics_results.json', 'w') as f:
        json.dump(stats_results, f, indent=4)

    for r in r_values:
        plt.figure(figsize=(10, 6))

        C_data = np.array(results['strategy_evolution'][r]['C'])
        D_data = np.array(results['strategy_evolution'][r]['D'])

        C_mean = np.mean(C_data, axis=0)
        C_std = np.std(C_data, axis=0)
        D_mean = np.mean(D_data, axis=0)
        D_std = np.std(D_data, axis=0)

        plt.plot(C_mean, 'b-', label='Cooperation (mean)')
        plt.fill_between(
            range(len(C_mean)),
            C_mean - C_std,
            C_mean + C_std,
            color='blue', alpha=0.2
        )

        plt.xlim(0, None)
        plt.xscale(
            'symlog',
            linthresh=1,
            linscale=0.5,
            subs=np.arange(1, 10)
        )

        if args.epochs == 2000:
            desired_xticks = [0, 10, 100, 1000, 2000]
            desired_xticklabels = ['0', '1e1', '1e2', '1e3', '2e3']
            plt.xticks(desired_xticks, desired_xticklabels, fontsize=fontsize)
        else:
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)

        plt.yticks(fra_yticks, fontsize=fontsize)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlabel('t', fontsize=fontsize, labelpad=10)
        plt.ylabel('Fractions', fontsize=fontsize, labelpad=10)
        plt.legend(loc='best', fontsize=fontsize)
        plt.ylim(0, 1)

        plt.savefig(
            f'{output_path}/strategy_evolution_r{r}_with_stats.png',
            dpi=300, bbox_inches='tight', pad_inches=0
        )

        plt.close()

    r_list = sorted(stats_results.keys())
    C_means = [stats_results[r]['C']['mean'] for r in r_list]
    C_stds = [stats_results[r]['C']['std'] for r in r_list]
    D_means = [stats_results[r]['D']['mean'] for r in r_list]
    D_stds = [stats_results[r]['D']['std'] for r in r_list]

    plt.figure(figsize=(10, 6))
    plt.errorbar(r_list, C_means, yerr=C_stds, fmt='bo', capsize=5, label='Cooperation')
    plt.xlabel('r', fontsize=fontsize + 4)
    plt.ylabel('Final Cooperation Fractions', fontsize=fontsize)
    plt.xticks(fontsize=fontsize + 4)
    plt.yticks(fontsize=fontsize + 4)
    plt.legend(fontsize=fontsize + 4)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(
        f'{output_path}/final_fractions_with_error_bars.png',
        format='png', dpi=300, bbox_inches='tight', pad_inches=0
    )

    plt.close()

    final_fractions_data = {
        'r_values': list(r_list),
        'C_means': [float(x) for x in C_means],
        'C_stds': [float(x) for x in C_stds]
    }

    with open(f'{output_path}/final_fractions_data.json', 'w') as f:
        json.dump(final_fractions_data, f, indent=4)

    if args.runs > 1:
        plot_cooperation_boxplot(stats_results, output_path, fontsize)
        plot_cooperation_violinplot(stats_results, output_path, fontsize)
    else:
        print("\nNote: Boxplot/Violinplot skipped because runs=1 (requires multiple runs)")

    print("\n=== Statistical Summary ===")
    for r in r_list:
        print(f"\nFor r = {r}:")
        print(f"Cooperation - Mean: {stats_results[r]['C']['mean']:.4f} ± {stats_results[r]['C']['std']:.4f}")
        print(f" 95% CI: [{stats_results[r]['C']['ci_lower']:.4f}, {stats_results[r]['C']['ci_upper']:.4f}]")
        print(f"Defection - Mean: {stats_results[r]['D']['mean']:.4f} ± {stats_results[r]['D']['std']:.4f}")
        print(f" 95% CI: [{stats_results[r]['D']['ci_lower']:.4f}, {stats_results[r]['D']['ci_upper']:.4f}]")

    print("\nAll experiments completed with statistical analysis!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('-epochs', type=int, default=10000, help='Epochs')
    parser.add_argument('-runs', type=int, default=1, help='Runs')
    parser.add_argument('-L_num', type=int, default=200, help='question size')
    parser.add_argument('-alpha', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.99, help='Gamma parameter')
    parser.add_argument('-clip_epsilon', type=float, default=0.2, help='Clip epsilon')
    parser.add_argument('-question', type=int, default=1, help='question')
    parser.add_argument('-ppo_epochs', type=int, default=1, help='PPO epochs')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('-device', type=str, default='cuda', help='Device')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-output_path', type=str, default='data', help='output path')
    parser.add_argument('-delta', type=float, default=0.5, help='delta')
    parser.add_argument('-rho', type=float, default=0.001, help='rho')

    parser.add_argument('-zeta', type=float, default=0.2, help='Quota ratio (0.0 means disabled)')
    parser.add_argument('-beta', type=float, default=2.0, help='Unified reward/punishment strength')

    args = parser.parse_args()
    asyncio.run(main(args))