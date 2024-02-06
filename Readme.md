# COMPASS: High-Efficiency Deep Image Compression with Arbitrary-scale Spatial Scalability

## ICCV 2023

### [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_COMPASS_High-Efficiency_Deep_Image_Compression_with_Arbitrary-scale_Spatial_Scalability_ICCV_2023_paper.pdf) | [Project Page](https://imjongminpark.github.io/compass_webpage/) | [Video](https://www.youtube.com/watch?v=Zfo3f__suwQ)

## Installation

COMPASS supports python 3.7+ and Pytorch 1.13+.

```bash
git clone https://github.com/ImJongminPark/COMPASS.git
cd COMPASS
conda create -n compass python=3.7 -y
pip install torch torchvision
```

### Requirements

* PyYAML
* tensorboard
* thop

## Datasets

You can download the training and test datasets via this [link](https://drive.google.com/drive/folders/18-H3ukaMlcqKjbtHxfMlq_cToesOkAo6).

```bash
mkdir datsets_img
mv <YOUR_DOWNLOAD_PATH>/train_512.zip datasets_img
mv <YOUR_DOWNLOAD_PATH>/test.zip datasets_img

cd datasets_img
unzip train_512.zip -d train_512
unzip test.zip -d test
```
## Training

Before the training process, download the pre-trained residual compression module and LIFF module via this [link](https://drive.google.com/file/d/12pDQtEWjM9NOnfqnlMs87M8rjHdg2eBi/view?usp=drive_link).

```bash
mkdir pretrained
mv <YOUR_DOWNLOAD_PATH>/pretrained.zip pretrained
cd pretrained
unzip pretrained.zip
```

For the training process, choose a lambda value from the set [0.0018, 0.0035, 0.0067, 0.013]. Then, assign this selected value to the 'lmbda' parameter within the 'cfg_train.yaml' configuration file. Ensure this lambda value is consistent with the pre-trained residual compression module you intend to use.

```bash
python -m torch.distributed.launch --nproc_per_node=<NUM_OF_GPUS> train.py
```

## Evaluation

Before the evaluation process, download the whole pre-trained COMPASS model via this [link](https://drive.google.com/file/d/1up8soOMn1tfcSWNW6rl2CknnOw6AuvuU/view?usp=drive_link).

```bash
mkdir checkpoints
mv <YOUR_DOWNLOAD_PATH>/checkpoints.zip checkpoints
cd checkpoints
unzip checkpoints.zip
```

For the evaluation process, choose a lambda value from the set [0.0018, 0.0035, 0.0067, 0.013]. Then, assign this selected value to the 'lmbda' parameter within the 'cfg_eval.yaml' configuration file.

```bash
python update.py
python eval.py
```

## Acknowledgements

This work was supported by internal fund/grant of Electronics and Telecommunications Research Institute (ETRI). [23YC1100, Technology Development for Strengthening Competitiveness in Standard IPR for communication and media]

## Authors

* Jongmin Park, Jooyoung Lee, and Mulchurl Kim

## Citation

If you use this project, please cite the relevant original publications for the
models and datasets, and cite this project as:

```
@inproceedings{park2023compass,
  title={COMPASS: High-Efficiency Deep Image Compression with Arbitrary-scale Spatial Scalability},
  author={Park, Jongmin and Lee, Jooyoung and Kim, Munchurl},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12826--12835},
  year={2023}
}
```

