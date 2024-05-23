## End-to-end Video Text Detection, Tracking and Recognition

## Introduction
This is CMPE258 Deep Learning group project, which aims to build a full pipeline of Deep Learning application with model training.

Group Members:
- Jing Shu, jing.shu@sjsu.edu, (015941146)
- Sarah Yu, kunhong.yu@sjsu.edu, (016099252)
- Adeel Javed, adeel.javed@sjsu.edu, (014183143)

## Installation
The codebases are built on top of [TransDETR](https://github.com/weijiawu/TransDETR/tree/main), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTR](https://github.com/megvii-model/MOTR).
- Linux, CUDA>= 9.2, GCC>=5.4, Python>=3.7
- Use Anaconda to create a conda environment:
```
conda create -n TransDETR2 python=3.9 pip
conda activate TransDETR
conda install pytorch=1.6.0 torchvision=0.6.1
conda install -c conda-forge cudatoolkit=11.8.0
```
- Other requirements
```
pip install -r requirements.txt
```
- Build MultiScaleDeformableAttention and Rotated ROIAlign
```
cd ./models/ops
sh ./make.sh

cd ./models/Rotated_ROIAlign
python setup.py build_ext --inplace
```
## Usage
### Dataset preparation
- Download the dataset: [COCOTextV2](https://bgshih.github.io/cocotext/ ), [ICDAR 2015](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=4), [DSText](https://rrc.cvc.uab.es/?ch=22&com=downloads)
- Extract frames from videos: run ExtractFrame_FromVideo.py to extract frames and organize the extracted images in a folder structure below:
  ![Pasted Graphic 4](https://github.com/JingShu2021/CMPE258_Team11/assets/98684620/01027e07-b1d0-454b-a49b-062445ae8132)
- Generated labels for images:
  - COCOTextV2: run gen_labels_COCOTextV2.py
  - ICDAR 2015: run gen_labels_ICDAR15_video.py
  - DSText: run gen_labels_DSText.py
### Training and evaluation
- Download COCOTextV2 pre-trained weights for Pretrained [TransDETR](https://drive.google.com/file/d/1PvOvBVpJLewN5uMnSeiJddmDGh3rKcyv/view), then as following:
```
sh configs/r50_TransDETR_pretrain_COCOText.sh
```
- Train on ICDAR2015 as follows:
```
sh configs/r50_TransDETR_train_ICDAR15video.sh
```
- Train on DSText as follows:
```
sh configs/r50_TransDETR_train_DSText.sh
```
- Evaluate on IDCAR2015
```
sh configs/r50_TransDETR_eval_ICDAR2015.sh
```
### Inference
```
python inference.py
```
### End-to-End Application
Install flask (pip install flask).
```
python app.py
```
## Citing

This code uses codes from TransDETR, MOTR, TransVTSpotter and EAST. Many thanks to their wonderful work:
```
@article{wu2022transdetr,
  title={End-to-End Video Text Spotting with Transformer},
  author={Weijia Wu, Chunhua Shen, Yuanqiang Cai, Debing Zhang, Ying Fu, Ping Luo, Hong Zhou},
  journal={arxiv},
  year={2022}
}

@inproceedings{zeng2021motr,
  title={MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author={Zeng, Fangao and Dong, Bin and Zhang, Yuang and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

@article{wu2021bilingual,
  title={A bilingual, OpenWorld video text dataset and end-to-end video text spotter with transformer},
  author={Wu, Weijia and Cai, Yuanqiang and Zhang, Debing and Wang, Sibo and Li, Zhuang and Li, Jiahong and Tang, Yejun and Zhou, Hong},
  journal={arXiv preprint arXiv:2112.04888},
  year={2021}
}

@inproceedings{zhou2017east,
  title={East: an efficient and accurate scene text detector},
  author={Zhou, Xinyu and Yao, Cong and Wen, He and Wang, Yuzhi and Zhou, Shuchang and He, Weiran and Liang, Jiajun},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={5551--5560},
  year={2017}
}

```
