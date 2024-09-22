import torch


def download_alexnet():
    model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
    model_path = f"onnx/alexnet.onnx"

    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        model_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
    print(f"Torch model alexnet.onnx successfully exported")


# Exported models should afterwards be ran through the tools/symbolic_shape_infer.py script
if __name__ == "__main__":
    download_alexnet()

    # TODO: try more sophisticated models yourself
