# === File: src/model/SVR/run.py ===
import argparse
from trainer import train_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="CN, US, EU")
    parser.add_argument("--horizon", type=int, required=True, help="1 (Q1) or 3 (Q3)")
    args = parser.parse_args()

    train_and_evaluate(dataset=args.dataset, horizon=args.horizon)