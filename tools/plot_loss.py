import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def read_loss_csv(path: str):
    steps, losses = [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="logs/loss.png")
    args = p.parse_args()

    steps, losses = read_loss_csv(args.csv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(Path(args.csv).name)
    plt.savefig(args.out, dpi=200)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()
