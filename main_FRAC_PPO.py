import torch
from FRAC_PPO import SPGG
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import asyncio
import argparse
import os
import shutil
import re


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
    r_values = [4.4]
    # r_values = [round(i * 0.1, 1) for i in range(30, 61)]

    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'mps':
        device = torch.device("mps")

    if args.epochs == 1000:
        xticks = [0, 1, 10, 100, 1000]
    elif args.epochs == 10000:
        xticks = [0, 1, 10, 100, 1000, 10000]
    elif args.epochs == 100000:
        xticks = [0, 1, 10, 100, 1000, 10000, 100000]

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

    if args.zeta > 0:
        output_path = f'data/PPO_{formatted_time}_Zeta_{args.zeta}_beta_{args.beta}_q{str(args.question)}_e_{args.epochs}_L_{args.L_num}'
    else:
        output_path = f'data/PPO_{formatted_time}_q{str(args.question)}_e_{args.epochs}_L_{args.L_num}_a_{args.alpha}_g_{args.gamma}_ce_{args.clip_epsilon}_gl_{args.gae_lambda}_p_{args.ppo_epochs}_b_{args.batch_size}_delta_{args.delta}_rho_{args.rho}_seed_{args.seed}_beta_{args.beta}'

    save_params_to_json(experiment_params, filename_prefix="params", output_path=output_path)

    for r in r_values:
        print(f"\nRunning experiment with r={r}")
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
            plt.xscale('symlog', linthresh=1, linscale=0.5, subs=np.arange(1, 10))
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)
            plt.yticks(fra_yticks, fontsize=fontsize)

            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.xlabel('t', fontsize=fontsize, labelpad=10)
            plt.ylabel('Fractions', fontsize=fontsize, labelpad=10)
            plt.legend(loc='best', fontsize=fontsize)

            plt.savefig(
                f'{output_path}/strategy_evolution_r{r}_run{num}.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()

    print("All experiments completed!")

    folder_path = f'{output_path}/Density_C'
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

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

    if x_values:
        plt.clf()
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker="s", markersize=10, markerfacecolor='none', linestyle="-", color="b", label='C')
        plt.plot(x_values, 1 - y_values, marker="^", markersize=10, markerfacecolor='none', linestyle="-", color="r", label='D')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.xlabel("r", fontsize=16)
        plt.ylabel("Fractions", fontsize=16)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(
            f'{output_path}/C_D_r.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()
    else:
        print("没有找到有效的数据。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('-epochs', type=int, default=1000, help='Epochs')
    parser.add_argument('-runs', type=int, default=1, help='Runs')
    parser.add_argument('-L_num', type=int, default=200, help='question size')
    parser.add_argument('-alpha', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.99, help='Gamma parameter')
    parser.add_argument('-clip_epsilon', type=float, default=0.2, help='Clip epsilon')
    parser.add_argument('-question', type=int, default=2, help='question')
    parser.add_argument('-ppo_epochs', type=int, default=1, help='PPO epochs')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('-device', type=str, default='cuda', help='Device')
    parser.add_argument('-seed', type=int, default=41, help='random seed')
    parser.add_argument('-output_path', type=str, default='data', help='output path')
    parser.add_argument('-delta', type=float, default=0.5, help='delta')
    parser.add_argument('-rho', type=float, default=0.001, help='rho')

    parser.add_argument('-zeta', type=float, default=0.2, help='Quota ratio (0.0 means disabled)')
    parser.add_argument('-beta', type=float, default=2.0, help='Unified reward/punishment strength')

    args = parser.parse_args()
    asyncio.run(main(args))