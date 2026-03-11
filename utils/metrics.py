import torch

def accuracy(outputs, labels, topk=(1,)):
    if isinstance(outputs[0], torch.Tensor):
        return single_accuracy(outputs, labels, topk)
    
    accs = []
    for out in outputs:
        accs.append(single_accuracy(out, labels, topk))

    return sum(accs)/len(accs)

    
def single_accuracy(output, target, topk=(1,)):
    _, pred = torch.max(output, 1)

    correct = (pred == target).sum().item()

    return correct/target.size(0)