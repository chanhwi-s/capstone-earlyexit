import torch
from utils.metrics import accuracy

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_acc = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    return total_loss/len(loader), total_acc/len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    return total_loss/len(loader), total_acc/len(loader)
