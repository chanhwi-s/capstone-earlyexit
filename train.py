import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models.resnet18 import resnet18
from models.resnet18_pt_ee import resnet18_pt_ee
from datasets.dataloader import get_dataloader
from engine.trainer import train_one_epoch, evaluate
from utils.experiment import create_experiment_dir
from utils.config import load_config
from utils.seed import set_seed


"""
config 불러와서 unpacking 진행
"""
config_path = "configs/train.yaml"
cfg = load_config(config_path)

# dataset config
dataset=cfg['dataset']["name"]
data_root=cfg['dataset']["data_root"]
num_workers=cfg['dataset']["num_workers"]

# train config
batch_size = cfg["train"]["batch_size"]
epochs = cfg["train"]["epochs"]
seed = cfg["train"]["seed"]
weights = (
    cfg["train"]["w1"],
    cfg["train"]["w2"],
    cfg["train"]["w3"],
)

# optimizer config
lr = float(cfg["optimizer"]["lr"])
momentum = float(cfg["optimizer"]["momentum"])
weight_decay = float(cfg["optimizer"]["weight_decay"])

def train():
    set_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    exp_dir = create_experiment_dir(config_path)

    writer = SummaryWriter(f"{exp_dir}/tensorboard")


    train_loader, test_loader, num_classes = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed
    )
    model = resnet18_pt_ee(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    for epoch in range(epochs):

        train_loss, train_acc1, train_acc2, train_acc3= train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        """
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )
        """

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train_loss={train_loss:.4f} "
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc ee1, ee2, ee3", train_acc1, train_acc2, train_acc3, epoch)
        #writer.add_scalar("test/acc", test_acc, epoch)

        torch.save(
            model.state_dict(),
            f"{exp_dir}/checkpoints/epoch_{epoch+1}.pth"
        )

    torch.save(
        model.state_dict(),
        f"{exp_dir}/checkpoints/final.pth"
    )

    writer.close()


if __name__ == "__main__":
    train()
