import json
import sys
import numpy as np

def parse_trt_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # data는 list
    latencies = []

    for item in data:
        if "startEnqMs" in item and "endEnqMs" in item:
            lat = item["endEnqMs"] - item["startEnqMs"]
            latencies.append(lat)

    latencies = np.array(latencies)

    print("=== TensorRT Latency Summary ===")
    print(f"Samples       : {len(latencies)}")
    print(f"Mean (ms)     : {latencies.mean():.4f}")
    print(f"Median (ms)   : {np.median(latencies):.4f}")
    print(f"Min (ms)      : {latencies.min():.4f}")
    print(f"Max (ms)      : {latencies.max():.4f}")
    print(f"P90 (ms)      : {np.percentile(latencies, 90):.4f}")
    print(f"P95 (ms)      : {np.percentile(latencies, 95):.4f}")
    print(f"P99 (ms)      : {np.percentile(latencies, 99):.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_trt_profile.py <json_path>")
        sys.exit(1)

    parse_trt_json(sys.argv[1])
    