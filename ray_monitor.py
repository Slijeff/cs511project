import signal
import subprocess
import sys
import time

import coremltools as ct
import numpy as np
import ray
import torch
from PIL import Image
from torch import nn

# Initialize Ray
ray.init()


@ray.remote
def increment(n):
    result = 0
    for i in range(n):
        result += 1
    return result


@ray.remote
def convolution_gpu(n):
    device = torch.device("mps")  # for M-series chips
    m = nn.Conv2d(16, 32, 3, stride=2, device=device)
    _ = m(torch.randn(64, 16, n, n, device=device))
    return _


@ray.remote
def convolution_cpu(n):
    m = nn.Conv2d(16, 32, 3, stride=2)
    _ = m(torch.randn(64, 16, n, n))
    return _


@ray.remote
def coreml_ane():
    # Load Core ML model
    model = ct.models.MLModel("ResNet18.mlpackage")

    # Preprocess image
    image = Image.open("example.jpg").resize((224, 224))
    input_data = np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.repeat(input_data, 8, axis=0)

    # Run inference
    result = model.predict({"x_1": input_data})
    return result
    # print(result)


# Start powermetrics in the background
def start_powermetrics(log_file, interval=500):
    cmd = [
        "sudo",
        "powermetrics",
        "--samplers",
        "cpu_power",
        "-i",
        str(interval),  # Interval in milliseconds
    ]
    with open(log_file, "w") as f:
        powermetrics_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


# Stop powermetrics
# Stop powermetrics
def stop_powermetrics():
    global powermetrics_process
    if powermetrics_process:
        print("terminating...")
        powermetrics_process.terminate()
        powermetrics_process.wait()


# Signal handler for graceful termination
def signal_handler(sig, frame):
    global powermetrics_process
    if powermetrics_process:
        print("\nStopping powermetrics...")
        powermetrics_process.terminate()
        powermetrics_process.wait()
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


# Parse powermetrics log for energy data
def parse_powermetrics_log(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
    cpu_power = []
    gpu_power = []
    ane_power = []

    for line in lines:
        if "CPU Power" in line:
            try:
                power = float(line.split(" ")[2])
                cpu_power.append(power)
            except (ValueError, IndexError):
                continue
        elif "GPU Power" in line:
            try:
                power = float(line.split(" ")[2])
                gpu_power.append(power)
            except (ValueError, IndexError):
                continue
        elif "ANE Power" in line:
            try:
                power = float(line.split(" ")[2])
                ane_power.append(power)
            except (ValueError, IndexError):
                continue
    cpu_avg = sum(cpu_power) / len(cpu_power) if cpu_power else 0
    gpu_avg = sum(gpu_power) / len(gpu_power) if gpu_power else 0
    ane_avg = sum(ane_power) / len(ane_power) if ane_power else 0
    print(f"#measurements: {len(cpu_power)}")

    # convert milli watt to watt
    return cpu_avg / 1000, gpu_avg / 1000, ane_avg / 1000


# Run the benchmark
def benchmark():
    log_file = "./powermetrics_log.txt"
    interval = 500  # Sampling interval in milliseconds

    print("Starting powermetrics...")
    # start_powermetrics(log_file, interval)
    cmd = [
        "sudo",
        "powermetrics",
        "--samplers",
        "cpu_power,gpu_power,ane_power",
        "-i",
        str(interval),  # Interval in milliseconds
    ]
    f = open(log_file, "w")
    powermetrics_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    time.sleep(1)  # Allow time for powermetrics to stabilize

    print("Running Ray tasks...")
    start_time = time.time()
    tasks = [convolution_gpu.remote(100) for _ in range(100)]
    ray.get(tasks)
    end_time = time.time()

    # stop_powermetrics()
    if powermetrics_process:
        print("Stopping powermetrics...")
        f.close()
        powermetrics_process.terminate()
        powermetrics_process.wait()

    print("Parsing powermetrics log...")
    cpu_power, gpu_power, ane_power = parse_powermetrics_log(log_file)

    duration = end_time - start_time
    print(f"Task duration: {duration:.2f} seconds")
    print(f"Average CPU power: {cpu_power:.2f} W")
    print(f"Average GPU power: {gpu_power:.2f} W")
    print(f"Average ANE power: {ane_power:.2f} W")
    total_energy = (cpu_power + gpu_power + ane_power) * duration
    print(f"Total energy consumption: {total_energy:.2f} Joules")


if __name__ == "__main__":
    benchmark()
