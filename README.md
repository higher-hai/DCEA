# DCEA

This repository contains the implementation of the paper **"DCEA: DETR With Concentrated Deformable Attention for End-to-End Ship Detection in SAR Images"**.

## Requirements

Ensure the following dependencies are installed:
- `python==3.8.0`
- `torch==2.0.1`
- `torchvision==0.15.2`
- `onnx==1.14.0`
- `onnxruntime==1.15.1`
- `pycocotools`
- `PyYAML`
- `scipy`

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset

To ensure seamless integration, prepare your dataset in the COCO standard format as outlined below.
1. Place the dataset in the following path: `configs/dataset/coco/`.
2. Structure the dataset files as follows:
```
coco/
  annotations/  # COCO annotation JSON files
  train2017/    # training images
  val2017/      # validation images
```

## Usage

### Training

To train the model, use:
```bash
python train.py -c path/to/config -r path/to/checkpoint
```
Replace `path/to/config` with the path to your configuration file, and `path/to/checkpoint` with the path to an existing checkpoint if resuming training (optional).

### Evaluation

To evaluate the model, run:
```bash
python train.py -c path/to/config -r path/to/checkpoint --test-only
```
Adding `--test-only` will run evaluation only, without further training.

### Inference

For inference, use:
```bash
python inference.py
```
Before running inference, configure `inference.py` with the correct paths and parameters as needed.

## License

This project is released under the MIT License.

## Acknowledgments

This implementation is based on [DETR](https://github.com/facebookresearch/detr.git) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR.git) frameworks. We thank the original authors for their contributions.
