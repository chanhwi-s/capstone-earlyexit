"""
Plain ResNet-18 전용 Trainer

출력이 1개인 일반 분류 모델에서 사용.
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Returns:
        avg_loss  : float
        train_acc : float
    """
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total      += labels.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """
    Returns:
        avg_loss : float
        test_acc : float
    """
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            total_loss += criterion(logits, labels).item()
            total      += labels.size(0)
            correct    += logits.argmax(1).eq(labels).sum().item()

    return total_loss / len(loader), correct / total
