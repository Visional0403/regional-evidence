# Trained weights

Download the assets from [weights-v1](https://github.com/Visional0403/regional-evidence/releases/tag/weights-v1).

- `qwen3b-{en,zh}.zip`, `qwen7b-{en,zh}.zip` and `qwen32b-{en,zh}.zip` contain `lora/` and `fusion.safetensors`. The latter contains the frozen PVTv2-B0/FPN and the trained fusion parameters.
- `kimi-{en,zh}.zip` contain the language-specific `lora/` and the matching `fusion.safetensors` trained with the native Kimi pixel normalisation. Use the same LoRA for the baseline and expert arms.
- `visdrone-pvtv2b0-fpn.safetensors` contains the shared epoch-120 frozen domain expert for fusion training.
- `qwen7b-{en,zh}-p1-trained.safetensors` contains the P1-value-trained models for the readout ablation. Use the matching 7B language adapter.

Built with Qwen. The Qwen2.5-VL-3B-derived assets are for research and evaluation under the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE), which accompanies the archives. Base model weights are distributed by the original authors under their own terms.
