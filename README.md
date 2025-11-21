<div align="center">

## Controllable Layer Decomposition for Reversible Multi-Layer Image Generation

</div>

-----

### 📢 News

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
@misc{liu2025controllablelayerdecompositionreversible,
    title={Controllable Layer Decomposition for Reversible Multi-Layer Image Generation}, 
    author={Zihao Liu and Zunnan Xu and Shi Shu and Jun Zhou and Ruicheng Zhang and Zhenchao Tang and Xiu Li},
    year={2025},
    eprint={2511.16249},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2511.16249}, 
}
```
