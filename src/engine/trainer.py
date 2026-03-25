import torch
from utils import accuracy

def train_one_epoch(model, loader, optimizer, criterion, device, weights=(0.3, 0.3, 1.0)):
    """
    EE 모델 / plain 모델 모두 대응.

    Returns:
        avg_loss : float
        acc1     : float | None  — EE 모델만 해당 (plain이면 None)
        acc2     : float | None  — EE 모델만 해당 (plain이면 None)
        acc3     : float         — 최종 출력 accuracy
    """
    model.train()

    total_loss = 0
    total      = 0
    correct1   = 0
    correct2   = 0
    correct3   = 0
    is_ee      = None   # 첫 배치에서 결정

    w1, w2, w3 = weights

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # 첫 배치에서 EE 모델 여부 판단
        if is_ee is None:
            is_ee = isinstance(outputs, list)

        if is_ee:
            out1, out2, out_final = outputs
            loss = w1 * criterion(out1, labels) \
                 + w2 * criterion(out2, labels) \
                 + w3 * criterion(out_final, labels)
        else:
            out1 = out2 = None
            out_final = outputs
            loss = criterion(out_final, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total      += labels.size(0)

        _, p3 = out_final.max(1)
        correct3 += p3.eq(labels).sum().item()

        if is_ee:
            _, p1 = out1.max(1)
            _, p2 = out2.max(1)
            correct1 += p1.eq(labels).sum().item()
            correct2 += p2.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc1 = correct1 / total if is_ee else None
    acc2 = correct2 / total if is_ee else None
    acc3 = correct3 / total

    return avg_loss, acc1, acc2, acc3


def evaluate(model, loader, criterion, device):
    """
    EE 모델 / plain 모델 모두 대응.

    Returns:
        avg_loss : float         — main 출력 기준 loss
        acc_exit1: float | None  — EE 모델만 해당 (plain이면 None)
        acc_exit2: float | None  — EE 모델만 해당 (plain이면 None)
        acc_main : float         — 최종 출력 accuracy
    """
    model.eval()

    total_loss = 0
    correct1 = correct2 = correct_main = 0
    total = 0
    is_ee = None   # 첫 배치에서 결정

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if is_ee is None:
                is_ee = isinstance(outputs, list)

            if is_ee:
                out1, out2, out_main = outputs
            else:
                out1 = out2 = None
                out_main = outputs

            loss = criterion(out_main, labels)
            total_loss += loss.item()

            _, p_main = out_main.max(1)
            correct_main += p_main.eq(labels).sum().item()

            if is_ee:
                _, p1 = out1.max(1)
                _, p2 = out2.max(1)
                correct1 += p1.eq(labels).sum().item()
                correct2 += p2.eq(labels).sum().item()

            total += labels.size(0)

    acc_exit1 = correct1 / total if is_ee else None
    acc_exit2 = correct2 / total if is_ee else None
    acc_main  = correct_main / total

    return total_loss / len(loader), acc_exit1, acc_exit2, acc_main
