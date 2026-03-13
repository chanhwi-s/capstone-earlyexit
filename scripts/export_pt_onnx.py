import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import torch
import importlib


def export_pt_onnx(model_name: str):

    try:
        module = importlib.import_module(f"src.models.{model_name}")
    except ModuleNotFoundError as e:
        print(e)
        print(f"Model file not found: src/models/{model_name}.py")
        sys.exit(1)

    if not hasattr(module, "build_model"):
        print(f"{model_name}.py must define build_model()")
        sys.exit(1)

    model = module.build_model(num_classes=1000)

    model_dir = os.path.join(BASE_DIR, "artifacts", model_name)
    os.makedirs(model_dir, exist_ok=True)

    pt_path = os.path.join(model_dir, "model.pt")
    onnx_path = os.path.join(model_dir, "model.onnx")

    torch.save(model.state_dict(), pt_path)
    print(f"{pt_path} 저장 완료")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    print(f"{onnx_path} 생성 완료")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_pt_onnx.py <model_name>")
        sys.exit(1)

    export_pt_onnx(sys.argv[1])