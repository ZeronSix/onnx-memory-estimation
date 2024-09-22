import argparse
from pathlib import Path

import onnx

from memory import DeviceMemory
from strategies import estimate_mutable_tensors_naive


def main(args: argparse.Namespace):
    model = onnx.load_model(args.model_path)

    max_symbolic_param_values = {
        "batch_size": 8,  # TODO: try different values
    }

    memory = DeviceMemory(alignment=args.alignment)

    if args.strategy == "naive":
        mutable_memory_size, mutable_tensors_info = estimate_mutable_tensors_naive(
            model, max_symbolic_param_values, memory
        )
    else:
        # TODO: test your strategies here
        raise NotImplementedError

    print("Mutable tensor estimation result:")
    for tensor_name, tensor_info in mutable_tensors_info.items():
        print(f"{tensor_name}: {tensor_info}")
    print(f"Total memory: {mutable_memory_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model with symbolic shapes",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=256,  # Default GPU alignment
        help="Memory alignment of tensors in bytes",
    )
    parser.add_argument("--strategy", choices=["naive"], default="naive")
    main(parser.parse_args())
