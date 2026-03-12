import torch
from src.models.resnet18 import resnet18
from src.models.resnet18_pt_ee import resnet18_pt_ee

model = resnet18(num_classes=1000)  # 또는 원하는 num_classes

def export_models():
    torch.save(model.state_dict(), '/artifactsresnet18_structure.pt')
    print("resnet18_structure.pt로 export되었습니다.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input':{0: 'batch_size'}, 'output':{0: 'batch_size'}},
        verbose=False
    )

    print('resnet onnx 생성 완료')

if __name__ == "__main__":
    export_models()