"""Verify PyTorch can see a CUDA GPU before running the pipeline."""

import sys

import torch


def main() -> int:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. Install the CUDA build of PyTorch or request a GPU node.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    for index in range(device_count):
        print(f"  [{index}] {torch.cuda.get_device_name(index)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
