<div align="center">

English | [简体中文](README_cn.md)

# Slim-DETR

**Slim-DETR: Real-time Tiny Object Detection with Efficient Interaction and Gaussian Query** (ECCV 2026)

An efficient tiny-object detector built on top of [D-FINE](https://github.com/Peterande/D-FINE), evaluated on [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) and AI-TOD-V2.

Official inference code, pretrained weights and training logs on VisDrone.

</div>

---

## 📋 Introduction

**Slim-DETR** is a real-time framework for tiny object detection with two core designs:

- **Efficient Interaction (EI) encoder** — a gather-and-inject mechanism that aggregates global semantics while restricting semantic attention interactions exclusively to extracted foreground tokens, drastically cutting the computational overhead of conventional Transformer-based encoders (implemented in [`src/zoo/dfine/hybrid_encoder_gather_sparseMSE_pan.py`](src/zoo/dfine/hybrid_encoder_gather_sparseMSE_pan.py), with a lightweight high-resolution PAN aggregation).
- **Gaussian Target Guided (GTG) query selection** — leverages the Gaussian spatial distribution of targets to dynamically concentrate high-quality queries on regions containing tiny objects (implemented in [`src/zoo/dfine/dfine_decoder_gauss.py`](src/zoo/dfine/dfine_decoder_gauss.py), with auxiliary Gaussian heatmap supervision).

Combined with a 4-layer decoder, Slim-DETR achieves a better accuracy / latency trade-off than D-FINE on VisDrone with **5.9M (R18) / 15.7M (R50) parameters**.

## 🏆 Results on VisDrone-val (input 640×640)

| Method | AP | AP<sub>50</sub> | AP<sub>75</sub> | Params(M) | GFLOPs | Latency(ms) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RT-DETR-R18 | 26.7 | 44.6 | 25.9 | 20.0 | 60 | 13.8 |
| D-FINE-R50 | 32.0 | 52.1 | 33.2 | 40.7 | 117 | 18.0 |
| **Slim-DETR-R18 (Ours)** | **32.4** | **52.5** | **33.5** | **5.9** | **51** | **14.3** |
| **Slim-DETR-R50 (Ours)** | **33.9** | **55.2** | **36.9** | **15.7** | **131** | **24.4** |

> Training logs are provided in [`logs/`](logs/) (`json-lines`, one epoch per line). The released checkpoints reproduce `AP 32.4` (R18) and `AP 35.1` (R50) on VisDrone-val under the exact config in this repo (see [Reproduce validation](#-reproduce-validation)).

## 🎁 Weights

| Model | AP | AP<sub>50</sub> | Params(M) | Download |
|:-:|:-:|:-:|:-:|:-:|
| Slim-DETR-R18 | 32.4 | 52.5 | 5.9 | [`weights/slim_detr_r18_visdrone.pth`](weights/slim_detr_r18_visdrone.pth) |
| Slim-DETR-R50 | 33.9 | 55.2 | 15.7 | [`weights/slim_detr_r50_visdrone.pth`](weights/slim_detr_r50_visdrone.pth) |

## ⚡ Quick Start

### 1. Setup

```bash
conda create -n slim-detr python=3.10 -y
conda activate slim-detr
pip install -r requirements.txt
```

> Requires `torchvision >= 0.15.2`.

### 2. Inference on images

```bash
# Slim-DETR-R50 on the bundled demo images
python inference.py -c configs/dfine/slim_detr_r50_visdrone.yml \
    -r weights/slim_detr_r50_visdrone.pth \
    -i assets/examples/ -o results/ -t 0.4

# Slim-DETR-R18 on a single image
python inference.py -c configs/dfine/slim_detr_r18_visdrone.yml \
    -r weights/slim_detr_r18_visdrone.pth \
    -i assets/examples/0000330_00801_d_0000804.jpg -o results/
```

Arguments:

| flag | description |
|:-:|:-|
| `-c` | model config (`configs/dfine/slim_detr_r18_visdrone.yml` / `slim_detr_r50_visdrone.yml`) |
| `-r` | checkpoint path |
| `-i` | a single image, an image folder, or a video (`mp4/avi/mov/mkv`) |
| `-o` | output directory (default `results/`) |
| `-t` | score threshold (default `0.4`) |
| `-d` | device (default `cuda:0`) |

### 3. Demo results

Visualized predictions on VisDrone-val images (left: Slim-DETR-R18, right: Slim-DETR-R50) are provided under [`results/`](results/).

<p align="center">
  <img src="results/slim_detr_r50/0000330_00801_d_0000804.jpg" width="48%">
  <img src="results/slim_detr_r18/0000330_00801_d_0000804.jpg" width="48%">
</p>

<p align="center">
  <img src="results/slim_detr_r50/0000295_02400_d_0000033.jpg" width="48%">
  <img src="results/slim_detr_r18/0000295_02400_d_0000033.jpg" width="48%">
</p>

## 📊 Reproduce validation

Prepare the VisDrone dataset in COCO format:

```text
/path/to/VisDrone/
├── train/
│   ├── images/
│   └── train.json
└── val/
    ├── images/
    └── val.json
```

Update the paths in [`configs/dataset/visdrone_detection.yml`](configs/dataset/visdrone_detection.yml), then:

```bash
# evaluate on VisDrone-val (single GPU)
python train.py -c configs/dfine/slim_detr_r50_visdrone.yml \
    -r weights/slim_detr_r50_visdrone.pth --test-only

# multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    -c configs/dfine/slim_detr_r50_visdrone.yml --seed=0
```

## 📁 Repository Structure

```text
Slim-DETR
├── inference.py                     # inference on images / folders / videos
├── train.py                         # training & validation entry
├── configs
│   ├── dataset/visdrone_detection.yml
│   └── dfine
│       ├── slim_detr_r18_visdrone.yml
│       ├── slim_detr_r50_visdrone.yml
│       └── include/
├── weights                          # released checkpoints
├── logs                             # training logs (json-lines, one epoch per line)
├── assets/examples                  # sample VisDrone-val images
├── results                          # visualized predictions
└── src                              # model / data / engine implementation
```

Key implementation files:

| component | file |
|:-:|:-|
| EI encoder (gather + inject, lightweight PAN) | [`src/zoo/dfine/hybrid_encoder_gather_sparseMSE_pan.py`](src/zoo/dfine/hybrid_encoder_gather_sparseMSE_pan.py) |
| GTG decoder (Gaussian-heatmap guided queries) | [`src/zoo/dfine/dfine_decoder_gauss.py`](src/zoo/dfine/dfine_decoder_gauss.py) |
| Gaussian heatmap GT generator | [`src/zoo/dfine/generate_ht_gt.py`](src/zoo/dfine/generate_ht_gt.py) |
| foreground token sampling | [`src/zoo/dfine/df_sampler.py`](src/zoo/dfine/df_sampler.py) |

## 📖 Citation

If you find Slim-DETR useful for your research or applications, please cite:

```bibtex
@inproceedings{wu2026slimdetr,
  title     = {Slim-DETR: Real-time Tiny Object Detection with Efficient Interaction and Gaussian Query},
  author    = {Wu, Tong and Zhang, Jiahao and Guan, Juntao and Lai, Rui},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This project is built upon [D-FINE](https://github.com/Peterande/D-FINE) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR). Thanks to their authors for the great open-source work.

## 📜 License

This repository is released under the [Apache 2.0 license](LICENSE).
