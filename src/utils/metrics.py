import torch

def accuracy(output, target):
    _, pred = torch.max(output, 1)

    correct = (pred == target).sum().item()

    return correct/target.size(0)
