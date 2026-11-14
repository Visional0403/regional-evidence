# Exterior context induces regional errors in vision-language models

Code and numerical source data for the visual encoding, regional perturbations and local expert studied in the paper. The released comparison uses native Qwen2.5-VL and Kimi-VL backbones, matched regional inputs and a shared domain expert trained on VisDrone.

## Setup

Use Python 3.12. For data preparation and numerical summaries:

```sh
pip install -e .
prepare-visdrone data/VisDrone2019/VisDrone2019-DET-val validation.json --language en
summarize-regional-evidence results
```

Download the [official datasets](datasets/README.md). The region selections are in `splits/`. Figure source files and the final five-task and value-scale tables are described in [results](results/README.md). The [training configurations](configs/) list the optimisation and decoding settings.

## Models

Original pretrained weights come from the model authors: [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct), and [Kimi-VL-A3B](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct).

The original-image inference entry point accepts the corresponding trained LoRA and fusion weights. Install the model dependencies for the backbone family before running on a CUDA device:

```sh
pip install -e '.[qwen]'
python -m regional_evidence.predict qwen7b validation.json data/VisDrone2019 lora.jsonl \
  --lora weights/qwen7b-en/lora
python -m regional_evidence.predict qwen7b validation.json data/VisDrone2019 predictions.jsonl \
  --lora weights/qwen7b-en/lora --fusion weights/qwen7b-en/fusion.safetensors
python -m regional_evidence.score validation.json lora.jsonl predictions.jsonl --protocol en-paired
```

Use `.[kimi]` in a separate environment for Kimi. Download and extract the [trained weights](docs/weights.md) under `weights/`. Use `--values all` for the all-level value readout in the ablation. The default P1 readout retains all keys in the shared softmax.

For five-task comparisons of the pretrained, LoRA and expert arms, pass all three prediction files with `--protocol en-three-stage` or `--protocol zh-three-stage`. Use `--protocol zh-paired` for Chinese LoRA-versus-expert predictions.

For regional experiments, set the image root to the directory above `VisDrone2019/` or `UA-DETRAC/`:

```sh
python -m regional_evidence.region_eval presence splits/visdrone_presence.json data regions.jsonl
python -m regional_evidence.transfer qwen7b splits/visdrone_presence.json data transfers.jsonl
python -m regional_evidence.region_eval full_image_control splits/visdrone_full_image_controls.json data full_image_controls.jsonl
```

Regional evaluation writes a `truncated` flag for each condition alongside its reply. A flag is true when generation reaches its token limit without ending in a configured stop token. Exclude flagged replies before parsing and require valid replies in both arms for paired counting or category scores. The transfer command emits `[reply, first_token_score, truncated]` for each generation. A truncated transfer reply cannot count as a successful answer flip.

## Training

Prepare the official VisDrone training and validation questions with `prepare-visdrone`. Domain-expert training uses the [original COCO-pretrained RetinaNet PVTv2-B0 checkpoint](https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth).

Kimi training requires [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) built for the installed PyTorch and CUDA versions. Kimi evaluation uses eager attention.

```sh
prepare-visdrone data/VisDrone2019/VisDrone2019-DET-train train.json --language en
python -m regional_evidence.train_detector data/VisDrone2019/VisDrone2019-DET-train coco.pth expert.safetensors
python -m regional_evidence.train qwen7b lora configs/english/qwen7b.json train.json data/VisDrone2019 weights/lora \
  --validation validation.json
python -m regional_evidence.train qwen7b fusion configs/english/qwen7b.json train.json data/VisDrone2019 weights/fusion \
  --lora weights/lora --expert expert.safetensors
```

Original code is provided under the [Apache License 2.0](LICENSE). Dataset and model terms are documented with their upstream sources.
