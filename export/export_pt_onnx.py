import os
import torch

from src.models.resnet18 import resnet18
from src.models.resnet18_pt_ee import resnet18_pt_ee


def export_pt_onnx():
    model = resnet18(num_classes=1000)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pt_dir = os.path.join(base_dir, "artifacts", "pytorch")
    onnx_dir = os.path.join(base_dir, "artifacts", "onnx")

    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    pt_path = os.path.join(pt_dir, "plain_resnet18.pt")
    onnx_path = os.path.join(onnx_dir, "plain_resnet18.onnx")

    # export to pt
    torch.save(model.state_dict(), pt_path)
    print(f"{pt_path} 저장 완료")

    # export to ONNX
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
        dynamo=True,   # 기본값이라 생략 가능
    )

    print(f"{onnx_path} 생성 완료")


if __name__ == "__main__":
    export_pt_onnx()