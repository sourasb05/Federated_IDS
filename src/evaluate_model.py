# evaluate_model.py
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def eval_global(model, loader, device):

    model.eval()

    totals = 0
    loss_sum = 0.0
    use_ce = True

    all_logits = []
    all_y = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X, y = batch[0], batch[1]
        else:
            return {'n': 0, 'loss': None, 'acc': None, 'f1': None, 'auc': None, 'cm': None}

        X = X.to(device)
        y = y.to(device)

        out = model(X)
        logits = out[0] if (isinstance(out, (list, tuple)) and len(out) > 0) else out

        try:
            loss = F.cross_entropy(torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits, y)
            loss_sum += float(loss.item()) * y.size(0)
        except Exception:
            use_ce = False

        totals += y.size(0)
        logits_tensor = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
        all_logits.append(logits_tensor.detach().cpu())
        all_y.append(y.detach().cpu())

    if totals == 0:
        return {'n': 0, 'loss': None, 'acc': None, 'f1': None, 'auc': None, 'cm': None}

    import torch as _torch
    logits = _torch.cat(all_logits, dim=0)
    y_true = _torch.cat(all_y, dim=0)

    preds = logits.argmax(dim=1)
    acc = (preds == y_true).float().mean().item()
    mean_loss = (loss_sum / totals) if use_ce else None

    f1 = None
    auc = None
    cm = None
    y_np = y_true.numpy()
    p_np = preds.numpy()
    # F1
    try:
        n_classes = logits.shape[-1]
        avg = "binary" if n_classes == 2 else "macro"
        f1 = f1_score(y_np, p_np, average=avg, zero_division=0)
    except Exception:
        f1 = None
    # AUC (binary only; needs both classes)
    if logits.shape[-1] == 2:
        try:
            probs1 = F.softmax(logits, dim=1)[:, 1].numpy()
            if (y_np == 0).any() and (y_np == 1).any():
                auc = roc_auc_score(y_np, probs1)
        except Exception:
            auc = None
    # Confusion matrix
    try:
        cm = confusion_matrix(y_np, p_np)
    except Exception:
        cm = None

    return {'n': totals, 'loss': mean_loss, 'acc': acc, 'f1': f1, 'auc': auc, 'cm': cm}
