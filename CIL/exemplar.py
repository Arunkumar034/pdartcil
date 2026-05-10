import torch
from torch.utils.data import DataLoader, Subset


def select_exemplars(backbone, dataset, m, device):
    """
    Paper-exact exemplar selection using herding (Section III-C).

    Args:
        backbone: Joint feature extractor Φ_t
        dataset: Dataset object with .samples attribute
        m: Number of exemplars per class
        device: cuda or cpu

    Returns:
        List of (image_tensor, label) tuples
    """

    backbone.eval()
    exemplars = []

    # Get unique class labels from dataset
    labels = sorted(set([s[1] for s in dataset.samples]))

    for label in labels:

        # ---------------------------------------------------
        # 1️⃣ Collect all samples of this class
        # ---------------------------------------------------
        indices = [i for i, s in enumerate(dataset.samples) if s[1] == label]

        if len(indices) == 0:
            continue

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False)

        features = []
        images = []

        # ---------------------------------------------------
        # 2️⃣ Extract normalized features
        # Paper: features must be L2 normalized
        # ---------------------------------------------------
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)

                f = backbone(x)
                f = torch.nn.functional.normalize(f, dim=1)

                features.append(f.cpu())
                images.append(x.cpu())

        features = torch.cat(features, dim=0)   # [N, D]
        images = torch.cat(images, dim=0)       # [N, C, H, W]

        # ---------------------------------------------------
        # 3️⃣ Compute class mean (x* in paper)
        # ---------------------------------------------------
        class_mean = features.mean(dim=0)
        class_mean = torch.nn.functional.normalize(class_mean, dim=0)

        # ---------------------------------------------------
        # 4️⃣ Herding selection (Paper Section III-C)
        # Select samples approximating class mean
        # ---------------------------------------------------
        selected = []
        running_sum = torch.zeros_like(class_mean)

        for k in range(min(m, features.size(0))):

            # Compute distances for all candidates
            candidate_mean = (running_sum.unsqueeze(0) + features) / (k + 1)
            distances = torch.norm(class_mean.unsqueeze(0) - candidate_mean, dim=1)

            # 🔥 Prevent selecting the same exemplar twice
            if len(selected) > 0:
                distances[selected] = float("inf")

            idx = torch.argmin(distances).item()

            selected.append(idx)
            running_sum += features[idx]

        # ---------------------------------------------------
        # 5️⃣ Store selected exemplars
        # ---------------------------------------------------
        for idx in selected:
            exemplars.append((images[idx].cpu(), label))

    return exemplars
