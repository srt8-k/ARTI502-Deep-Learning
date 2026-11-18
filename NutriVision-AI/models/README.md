# Model weights (not included)

Copy these two existing, independently trained checkpoints from the team's approved private storage into this folder:

- dinov2_last4_best.pth (DINOv2, Food-101 101-class classification, 280 px)
- nutrition_mlp_best.pth (six-output nutrition head and training normalization)

The backend checks that both files exist and loads them with torch.load(weights_only=True). It does not download the team's checkpoints automatically.

SigLIP2 (google/siglip2-base-patch16-256) is downloaded separately through Hugging Face on first startup when NUTRIVISION_OPEN_VOCAB=1. Consider license, network access, CPU memory, and deployment costs.

Do not commit weights, cached model blobs, private image uploads, or feedback databases to GitHub. Check distribution permissions before sharing weights externally.
