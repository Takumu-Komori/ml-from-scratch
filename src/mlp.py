
# src/mlp.py
import argparse
import os
import csv
import time
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_regression(n: int, d: int, noise_std: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    true_w = torch.randn(d, 1, generator=g)
    true_b = torch.randn(1, generator=g)
    y = X @ true_w + true_b + noise_std * torch.randn(n, 1, generator=g)
    return X, y, true_w, true_b

class MLP(torch.nn.Module):
  def __init__(self, d_in: int, d_hidden: int, d_out: int):
    super().__init__()
    # parameters as learnable tensors
    self.fc1 = torch.nn.Linear(d_in, d_hidden)
    self.act = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(d_hidden, d_out)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    h = self.fc1(x)
    h = self.act(h)
    y = self.fc2(h)
    return y


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


def save_log_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--d_hidden", type=int, default=32)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_path", type=str, default="logs/mlp_loss.csv")
    args = p.parse_args()

    set_seed(args.seed)

    X, y, true_w, true_b = make_synthetic_regression(
        n=args.n, d=args.d, noise_std=args.noise_std, seed=args.seed
    )

    model = MLP(d_in=args.d, ad_hidden=rgs.d_hidden, d_out=1)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    log_rows = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        y_hat = model(X)
        loss = mse(y_hat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == 1:
            log_rows.append([step, float(loss.item())])

    save_log_csv(args.log_path, log_rows)

    # print summary
    dt = time.time() - t0
    print(f"Done in {dt:.2f}s")
    
    print(f"final loss: {float(loss.item()):.6f}")
    print("true    w:", true_w.detach().view(-1).tolist())
    print("true    b:", float(true_b.detach().item()))
    print(f"loss log saved to: {args.log_path}")


if __name__ == "__main__":
    main()
