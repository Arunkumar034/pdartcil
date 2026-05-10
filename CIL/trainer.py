import torch
import torch.nn.functional as F


def train_representation(
    backbone,
    classifier,
    aux_classifier,
    loader,
    optimizer,
    device,
    task_start_idx,
    lambda_a
):

    backbone.train()
    classifier.train()
    aux_classifier.train()

    total_loss = 0.0
    num_classes = classifier.fc.out_features

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # =========================
        # Extract branch features separately
        # =========================
        branch_outputs = []
        for extractor in backbone.extractors:
            branch_outputs.append(extractor(x))

        # Concatenate raw features
        joint_features = torch.cat(branch_outputs, dim=1)

        # Normalize ONLY for main classifier
        features = F.normalize(joint_features, dim=1)

        # =========================
        # Main Classification Loss (Eq. 10)
        # =========================
        logits = classifier(features)
        loss_cls = F.cross_entropy(logits, y)

        # =========================
        # Auxiliary Classification Loss (Eq. 11)
        # =========================
        new_branch_features = branch_outputs[-1]  # true F_t(x)

        aux_logits = aux_classifier(new_branch_features)

        y_aux = torch.zeros_like(y)
        mask = y >= task_start_idx
        y_aux[mask] = (y[mask] - task_start_idx) + 1

        loss_aux = F.cross_entropy(aux_logits, y_aux)

        # =========================
        # Expandable Representation Loss (Eq. 12)
        # =========================
        loss = loss_cls + lambda_a * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_classifier_tuning(
    backbone,
    classifier,
    loader,
    optimizer,
    device,
    T=2.0,
    task_classes=None
):

    backbone.eval()
    classifier.train()

    total_loss = 0.0
    num_classes = classifier.fc.out_features

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = backbone(x)

        logits = classifier(features)

        if task_classes is not None:
            logits = logits[:, :task_classes]

        loss = F.cross_entropy(logits / T, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
