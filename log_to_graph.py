from math import log
import os
from turtle import st
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(palette="twilight_shifted")


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
    all_powers = []  # a 2d list of powers
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
        all_powers.append(powers)
    ray = all_powers[1]
    no_ray = all_powers[0]
    # create the figure and axes
    fig, ax = plt.subplots()
    sns.regplot(x=times[:len(ray)], y=ray, ax=ax,
                label="Ray", order=3, fit_reg=True)
    sns.regplot(x=times[:len(no_ray)], y=no_ray,
                ax=ax, label="No Ray", order=3, fit_reg=True)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Total Power (Watts)")
    ax.set_title("Power Consumption (Batch Size = 2, Tasks = 200)")
    ax.legend()
    plt.savefig("./out/power_consumption_comparison.png", dpi=300)
    plt.clf()
    # plt.show()


if __name__ == "__main__":
    log_file = "./out/powermetrics_log_ray.txt"
    logs_to_ray_comparison("./out")
    # log_to_graph(log_file)
