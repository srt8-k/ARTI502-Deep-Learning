# Integration notebooks

Only the short [API_Client_Demo.ipynb](API_Client_Demo.ipynb) is shared with the team.

It shows how to connect to an already-running NutriVision FastAPI server,
upload a local food image, read the returned JSON, and submit a user-confirmed correction.

The full 133-code-cell training/research Colab notebook is intentionally **not included**.
It belongs in the project's private Google Drive archive, not the collaborative integration folder.

**Important:** This client demo does not train or load DINOv2, SigLIP2, or Nutrition MLP.
The API server loads the trained checkpoints independently. Its URL must be reachable
from the computer running the notebook (127.0.0.1 only works for the same machine).
