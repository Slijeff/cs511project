from math import log
import os
from turtle import st
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(palette="Set2")


def log_to_graph(log_file, interval=500):
    powers = []
    with open(log_file, "r") as f:
        for line in f:
            if "Combined Power (CPU + GPU + ANE):" in line:
                power = float(line.split(":")[1].strip().split(" ")[0])
                powers.append(power / 1000)
    times = list(range(0, len(powers) * interval, interval))
    times = [t / 1000 for t in times]
    p = sns.regplot(x=times, y=powers, order=3)
    p.set_xlabel("Time (Seconds)")
    p.set_ylabel("Total Power (Watts)")
    p.set_title("Power Consumption")
    plt.savefig("./out/power_consumption.png", dpi=300)
    plt.clf()
    # plt.show()


def logs_to_ray_comparison(log_directory, interval=500):
    # Read all log files in the directory
    log_files = [f for f in os.listdir(log_directory) if f.endswith(".txt")]
    all_powers = {}
    times = []  # take the max time from all the log files
    for log_file in log_files:
        powers = []
        with open(os.path.join(log_directory, log_file), "r") as f:
            for line in f:
                if "Combined Power (CPU + GPU + ANE):" in line:
                    power = float(line.split(":")[1].strip().split(" ")[0])
                    powers.append(power / 1000)
        cur_times = list(range(0, len(powers) * interval, interval))
        cur_times = [t / 1000 for t in cur_times]
        if len(cur_times) > len(times):
            times = cur_times
        all_powers[log_file] = powers
    ray = all_powers["powermetrics_log_ray.txt"]
    no_ray = all_powers["powermetrics_log_no_ray.txt"]
    ray_gpu = all_powers["powermetrics_log_ray_gpu.txt"]
    no_ray_gpu = all_powers["powermetrics_log_no_ray_gpu.txt"]
    # create the figure and axes
    fig, ax = plt.subplots()
    common_args = dict(ax=ax, order=3, fit_reg=True,
                       ci=50, scatter_kws={"alpha": 0.2})
    sns.regplot(x=times[:len(ray)], y=ray,
                label="Ray (CPU)", **common_args)
    sns.regplot(x=times[:len(no_ray)], y=no_ray,
                label="No Ray (CPU)", **common_args)
    sns.regplot(x=times[:len(ray_gpu)], y=ray_gpu,
                label="Ray (GPU)", **common_args)
    sns.regplot(x=times[:len(no_ray_gpu)], y=no_ray_gpu,
                label="No Ray (GPU)", **common_args)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Total Power (Watts)")
    ax.set_title("Power Consumption by Time (Batch Size = 2, Tasks = 200)")
    ax.legend()
    plt.savefig("./out/power_consumption_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.clf()
    # plt.show()


if __name__ == "__main__":
    log_file = "./out/powermetrics_log_ray.txt"
    logs_to_ray_comparison("./out")
    # log_to_graph(log_file)
