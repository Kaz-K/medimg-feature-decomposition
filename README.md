# medimg-feature-decomposition (Medical Image Analysis, 2021)

This is an official repository dedicated to the implementation of the techniques described in our paper <b>"Decomposing Normal and Abnormal Features of Medical Images for Content-Based Image Retrieval of Glioma Imaging."</b> Please refer to the <b>[paper](https://www.sciencedirect.com/science/article/pii/S1361841521002723)</b> for more details.

### License

The code in this repository is released under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Setup

Here we describe the setup required for the model training.

### Requirements

The following environment has been tested for reproducibility:

- Python 3.7
- CUDA 10.0
- [PyTorch](https://pytorch.org/) 1.6.0
- [Pytorch lightning](https://www.pytorchlightning.ai/index.html) 1.0.0

### Downloading datasets

Download the BraTS 2019 datasets from the [MICCAI 2019 project page](https://www.med.upenn.edu/cbica/brats-2019/) and organize them according to their types to align with the directory structure below.

- `./data/MICCAI_BraTS_2019_Data_Training/HGG`
- `./data/MICCAI_BraTS_2019_Data_Training/LGG`
- `./data/MICCAI_BraTS_2019_Data_Testing`
- `./data/MICCAI_BraTS_2019_Data_Validation`

### Preprocessing the datasets

Execute `preprocess.py` to perform the preprocessing on the dataset. This step primarily involves resizing the images to $256 \times 256$, conducting Z-score normalization at the volume level, and reassigning the labels as follows:

- 0: Background
- 1: Non-enhancing tumor core
- 2: Peritumoral edema
- 3: GD-enhancing tumor

After this process, datasets are created with each original volume divided into slices. These datasets will be output in the `./data` folder, for example as `./data/MICCAI_BraTS_2019_Data_Training_Slices/HGG`. These preprocessed datasets will then be used to run scripts for model training and evaluation.

### Running training

The configuration file for training the baseline model is `base.json`.
Please note that the filenames of the original BraTS datasets and the train/val/test dataset splits used in this experiment are different (refer to the paper for the reasons behind this). To execute training, you can specify this file and use the command `python run_model.py -c base.json -t`.

### Citation

If you find this code useful for your research, please cite our paper:

```
@article{Kobayashi2021,
  author = {Kazuma Kobayashi and Ryuichiro Hataya and Yusuke Kurose and Mototaka Miyake and Masamichi Takahashi and Akiko Nakagawa and Tatsuya Harada and Ryuji Hamamoto},
  doi = {10.1016/J.MEDIA.2021.102227},
  issn = {1361-8415},
  journal = {Medical Image Analysis},
  month = {12},
  pages = {102227},
  pmid = {34543911},
  publisher = {Elsevier},
  title = {Decomposing normal and abnormal features of medical images for content-based image retrieval of glioma imaging},
  volume = {74},
  year = {2021},
}
```
