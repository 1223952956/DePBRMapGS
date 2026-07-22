def normal_error(label, predict, mask=None):
    # H × W
    dot = 1 - (label * predict).sum(dim=0)

    # H × W
    diff = ((label - predict) ** 2).mean(dim=0)

    error = dot + diff

    if mask is None:
        return error.mean()

    mask = mask.squeeze().bool()

    if not mask.any():
        return error.sum() * 0.0

    return error[mask].mean()

def l2(label, predict):
    return ((label - predict) ** 2).mean()


def l1(label, predict):
    return (label - predict).abs().mean()
