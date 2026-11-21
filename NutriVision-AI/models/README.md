# Private release checkpoints

The V3 CNN demo requires all four files listed in the [startup guide](../README.md#run-on-windows). Obtain them from the team privately and keep their filenames. Set `NUTRIVISION_MODEL_DIR` to their shared directory or place them here. The API loads the released weights with strict state dictionaries. Experimental nutrition heads have a different layout and are not drop-in replacements.

SigLIP2 is optional and uses the separate `google/siglip2-base-patch16-256` model when enabled.
