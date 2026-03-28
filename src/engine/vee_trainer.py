"""
Very Early Exit ResNet-18 전용 Trainer

출력이 2개 (ee1, main) 인 VEE 모델에서만 사용.
가중 합산 손실:  loss = w1·L(ee1) + w2·L(main)
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device,
                    weights=(0.3, 1.0)):
    """
    Returns:
        avg_loss      : float
        train_acc_ee1 : float
        train_acc_main: float
    """
    model.train()
    w1, w2 = weights

    total_loss = 0.0
    correct1 = correct_main = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out_ee1, out_main = model(images)

        loss = (w1 * criterion(out_ee1,  labels)
              + w2 * criterion(out_main, labels))
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total        += labels.size(0)
        correct1     += out_ee1.argmax(1).eq(labels).sum().item()
        correct_main += out_main.argmax(1).eq(labels).sum().item()

    n = len(loader)
    return total_loss / n, correct1 / total, correct_main / total


def evaluate(model, loader, criterion, device):
    """
    Returns:
        avg_loss      : float  (main 출력 기준)
        test_acc_ee1  : float
        test_acc_main : float
    """
    model.eval()

    total_loss = 0.0
    correct1 = correct_main = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            out_ee1, out_main = model(images)

            total_loss   += criterion(out_main, labels).item()
            total        += labels.size(0)
            correct1     += out_ee1.argmax(1).eq(labels).sum().item()
            correct_main += out_main.argmax(1).eq(labels).sum().item()

    n = len(loader)
    return total_loss / n, correct1 / total, correct_main / total
