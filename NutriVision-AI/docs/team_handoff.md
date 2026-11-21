# V3 demonstration handoff

Use the [Windows startup guide](../README.md#run-on-windows) for the checkpoint set and both terminals. The four released model files are available in the [v3.0-submission release asset](https://github.com/srt8-k/ARTI502-Deep-Learning/releases/tag/v3.0-submission); download and extract them into `models/`. The presenter can confirm the active food detector and optional suggestion mode at `/health`.

Demonstrate one recognized meal, one uncertain regional dish if SigLIP2 is enabled, a no-food image, and a food-name correction. Feedback is stored in `.local/feedback.sqlite3` by default; set `NUTRIVISION_FEEDBACK_DB` to choose another private database.

For professor review, source code for model training is indexed at [training/README.md](../training/README.md). The stages document the historical experiments; do not promise a one-click clean Colab rerun, particularly for the Food Presence V2 hard-negative dataset dependencies.

Use the [pilot report](real_world_evaluation_v3.md) for examples and the [historical evaluation](evaluation_results.md) for benchmark context. The paired release-versus-density results in the [project README](../README.md#measured-results-and-limits) explain the deployed regression choice.
