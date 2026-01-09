def normal_error(label, predict):
    dot = (1 - (label * predict).sum(dim=0)).mean()
    diff = l2(label, predict)
    return dot + diff


def l2(label, predict):
    return ((label - predict) ** 2).mean()


def l1(label, predict):
    return (label - predict).abs().mean()
