# GenWM-Bench

A unified benchmark for generative image watermarking: embedding, extraction, robustness attacks and quality evaluation.

## Install

```bash
pip install -r requirements.txt
```

## Pretrained Weights & Datasets

Pretrained weights and datasets are hosted separately on Hugging Face:

- Model weights: https://huggingface.co/EXP-REPO/GenWM-Bench-models
- Datasets:
  - https://cocodataset.org/#download
  - https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts
  - https://huggingface.co/datasets/poloclub/diffusiondb

Download and place the weights under `data/models/<MethodName>/` before running.

## Usage

Run an experiment with a config file:

```bash
python main.py --config configs/experiments/<config_name>.yaml
```

Experiment configs are under `configs/experiments/`, per-method settings under `configs/methods/`, metric settings under `configs/metrics/`.

Step-by-step examples for each method are available as notebooks under `Experiments/`.
