"""
EE-ViT 전용 Trainer

12개 exit head 각각의 CrossEntropyLoss 가중 합산.
Backbone이 frozen이므로 exit head 파라미터만 업데이트됨.

가중치 모드 (--weight-mode):
  equal  (기본) : 모든 exit 동일 가중치 1.0
  linear        : exit i에 (i+1)/12 선형 증가 가중치 (초반 exit 낮게, 후반 높게)
"""

import torch


def _make_weights(mode: str, n_exits: int) -> list:
    if mode == "linear":
        return [(i + 1) / n_exits for i in range(n_exits)]
    # equal
    return [1.0] * n_exits


def train_one_epoch(model, loader, optimizer, criterion, device,
                    weight_mode: str = "equal"):
    """
    Returns:
        avg_loss     : float   (전체 가중 손실 평균)
        acc_per_exit : list[float]  길이 12, 각 exit head의 train accuracy
    """
    model.train()

    n_exits  = model.NUM_BLOCKS          # 12
    weights  = _make_weights(weight_mode, n_exits)
    w_sum    = sum(weights)

    total_loss       = 0.0
    correct_per_exit = [0] * n_exits
    total            = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits_list = model(images)     # list of 12 tensors [B, num_classes]

        loss = sum(
            weights[i] * criterion(logits_list[i], labels)
            for i in range(n_exits)
        ) / w_sum                       # normalize so scale ≈ single CE loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total      += labels.size(0)
        for i, logits in enumerate(logits_list):
            correct_per_exit[i] += logits.argmax(1).eq(labels).sum().item()

    n = len(loader)
    return (
        total_loss / n,
        [c / total for c in correct_per_exit],
    )


def evaluate(model, loader, criterion, device):
    """
    Returns:
        avg_loss     : float          (마지막 exit 기준)
        acc_per_exit : list[float]    길이 12
    """
    model.eval()

    n_exits          = model.NUM_BLOCKS
    total_loss       = 0.0
    correct_per_exit = [0] * n_exits
    total            = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits_list = model(images)

            total_loss += criterion(logits_list[-1], labels).item()
            total      += labels.size(0)
            for i, logits in enumerate(logits_list):
                correct_per_exit[i] += logits.argmax(1).eq(labels).sum().item()

    n = len(loader)
    return (
        total_loss / n,
        [c / total for c in correct_per_exit],
    )
