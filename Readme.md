# COMPASS: High-Efficiency Deep Image Compression with Arbitrary-scale Spatial Scalability

### [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_COMPASS_High-Efficiency_Deep_Image_Compression_with_Arbitrary-scale_Spatial_Scalability_ICCV_2023_paper.pdf) | [Project Page](https://imjongminpark.github.io/compass_webpage/) | [Video](https://www.youtube.com/watch?v=Zfo3f__suwQ) | [Data](https://drive.google.com/drive/folders/18-H3ukaMlcqKjbtHxfMlq_cToesOkAo6)

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

## Datasets

You can download the training and test datasets via this [link](https://drive.google.com/drive/folders/18-H3ukaMlcqKjbtHxfMlq_cToesOkAo6).

```bash
mkdir datsets_img
mv <DOWNLOAD_PATH>/train_512.zip datasets_img

cd datasets_img
unzip train_512.zip -d train_512
unzip test.zip -d test
```
## Training

```bash
python -m torch.distributed.launch --nproc_per_node=<NUM_OF_GPUS> train.py
```

## Tests

TBD

## License

TBD

## Contributing

TBD

## Acknowledgements


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

