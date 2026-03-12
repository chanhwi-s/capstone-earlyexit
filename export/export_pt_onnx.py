import os
import torch

from src.models.resnet18 import resnet18
# from src.models.resnet18_pt_ee import resnet18_pt_ee

def export_pt_onnx():
    model = resnet18(num_classes=1000)
    model_name = "plain_resnet18"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "artifacts", model_name)
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
    export_pt_onnx()
