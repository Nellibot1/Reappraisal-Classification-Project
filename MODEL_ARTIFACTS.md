# Model artifacts

Trained model weights are intentionally excluded from GitHub by `.gitignore`.

The local project currently contains 26 large `.pt` and `.safetensors` files totaling approximately 27.76 GB. These files exceed ordinary GitHub file limits and the total included Git LFS storage on a standard individual plan. Metrics, predictions, training histories, manifests, tokenizer configuration, and notebooks remain suitable for the Git repository.

## Local artifact locations

- `Code/BERT_models/experiments/baseline/`: baseline BERT and ModernBERT checkpoints
- `Code/BERT_models/experiments/lr_epoch_tuning/`: tuning checkpoints
- `Code/BERT_models/experiments/tapt/masked_language_modeling/`: task-adapted RoBERTa encoders
- `Code/BERT_models/experiments/tapt/downstream_comparison/`: downstream comparison checkpoints

## Recommended publication workflow

Upload selected final checkpoints to a dedicated model or research-artifact service such as Hugging Face, OSF, or Zenodo. Record the permanent URL, version, license, file size, and SHA-256 checksum here. Keep experimental/intermediate checkpoints local unless they are necessary for reproducing a reported result.

Do not commit checkpoint files directly to this repository. A future artifact release should include explicit download and verification instructions.
