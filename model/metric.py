import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target)
    return correct / len(target)

def sensitivity(output, target, task=None):
    """
           Correctly predicted adventitious events
    SE =  -----------------------------------------
           Total number of adventitious events
    """
    if task is None:
        raise RuntimeError
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        is_adv = target >= task
        # Task 1 does not have PQ, therefore Normal is labeled as 0
        # Task 2 has PQ, labeled at 1, which we also want to drop
        correct = torch.sum((pred == target)[is_adv])
    return correct / torch.sum(is_adv)

def sensitivity_task1(output, target):
    return sensitivity(output, target, task=1)

def sensitivity_task2(output, target):
    return sensitivity(output, target, task=2)

def specificity(output, target):
    """
           Correctly predicted normal events
    SP =  -----------------------------------------
           Total number of normal events
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        is_normal = target == 0
        correct = torch.sum((pred == target)[is_normal])
    return correct / torch.sum(is_normal)

def score(output, target, se_fn):
    se = se_fn(output, target)
    sp = specificity(output, target)
    avg_score = (se + sp) / 2
    har_score = (2*se*sp) / (se + sp)
    return (avg_score + har_score) / 2

def score_task1(output, target):
    return score(output, target, sensitivity_task1)

def score_task2(output, target):
    return score(output, target, sensitivity_task2)
