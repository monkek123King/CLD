<div align="center">

## Controllable Layer Decomposition for Reversible Multi-Layer Image Generation

🏠 [Homepage](https://monkek123King.github.io/CLD_page)      📄 [Paper](http://arxiv.org/abs/2511.16249)      🤗 [HuggingFace](https://huggingface.co/papers/2511.16249)


</div>


### 📢 News

  * **`Dec 2025`:** Experiment checkpoints are released [here](https://huggingface.co/thuteam/CLD)\! 🎉
  * **`Nov 2025`:** The paper is now available on [arXiv](https://arxiv.org/abs/2511.16249). ☕️

-----

## 🚀 Getting Started

### 🔧 Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda env create -f environment.yml
conda activate CLD
```

**b. Clone CLD.**
```
git clone https://github.com/monkek123King/CLD.git
```

### 📦 Prepare model ckpt
**a. Download FLUX.1-dev weights**
```
from huggingface_hub import snapshot_download

repo_id = "black-forest-labs/FLUX.1-dev"
snapshot_download(repo_id, local_dir=Path_to_pretrained_FLUX_model)
```

**b.Download adapter pre-trained weights**
```
from huggingface_hub import snapshot_download

repo_id = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"
snapshot_download(repo_id, local_dir=Path_to_pretrained_FLUX_adapter)
```

**c. Download LoRA weights for CLD from https://huggingface.co/thuteam/CLD**
```
ckpt
├── decouple_LoRA
│   ├── adapter
│   │   └── pytorch_lora_weights.safetensors
│   ├── layer_pe.pth
│   └── transformer
│       └── pytorch_lora_weights.safetensors
├── pre_trained_LoRA
│   └── pytorch_lora_weights.safetensors
├── prism_ft_LoRA
│   └── pytorch_lora_weights.safetensors
└── trans_vae
    └── 0008000.pt
```

**d. YAML configuration file**
```
pretrained_model_name_or_path: Path_to_pretrained_FLUX_model
pretrained_adapter_path: Path_to_pretrained_FLUX_adapter
transp_vae_path: "ckpt/trans_vae/0008000.pt"
pretrained_lora_dir: "ckpt/pre_trained_LoRA"
artplus_lora_dir: "ckpt/prism_ft_LoRA"
lora_ckpt: "ckpt/decouple_LoRA/transformer"
layer_ckpt: "ckpt/decouple_LoRA"
adapter_lora_dir: "ckpt/decouple_LoRA/adapter"
```


### 🏋️ Train and Evaluate

**Train**

```
python -m train.train -c train/train.yaml
```

**Infer**
```
python -m infer.infer -c infer/infer.yaml
```

**Eval**

Prepare the ground-truth samples.
```
python -m eval.prepare_gt
```

Evaluate to obtain the metric results.
```
python evaluate.py --pred-dir "Path_to_predict_results" --gt-dir "Path_to_gt_samples" --output-dir "Path_to_save_eval_results"
```

-----

## ✍️ Citation

If you find our work useful for your research, please consider citing our paper and giving this repository a star 🌟.

```bibtex
@article{liu2025controllable,
  title={Controllable Layer Decomposition for Reversible Multi-Layer Image Generation},
  author={Liu, Zihao and Xu, Zunnan and Shu, Shi and Zhou, Jun and Zhang, Ruicheng and Tang, Zhenchao and Li, Xiu},
  journal={arXiv preprint arXiv:2511.16249},
  year={2025}
}
```
