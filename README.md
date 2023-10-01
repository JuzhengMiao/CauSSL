# CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation (ICCV 2023)

### Introduction

We provide the codes for CPSCauSSL and MCCauSSL with the 3D V-Net architecture targeted for the Pancreas-CT Dataset.
### Requirements

1. Pytorch
2. TensorBoardX
3. Some basic python packages such as Numpy

### Usage
1. Data preprocessing:
   
   We follow the same preprocessing pipeline of "Inconsistency-aware uncertainty estimation for semi-supervised medical image segmentation" (https://github.com/koncle/CoraNet).

   We also provide the data split files in the "Pancreas-CT" folder.

   Please remember to change related paths in the codes.
   
2. Train the model:

   python train_CT_CPSCauSSL.py
   
   python train_CT_MCCauSSL.py

3. Test the model:

   For the CPSCauSSL method, the testing has been included in "train_CT_CPSCauSSL.py".

   For the MCCauSSL method: python test_CT_norm_mct.py

### Acknowledgement
This code is based on the framework of UA-MT. We thank the authors for their codebase.

## Citation
If you find the code useful for your research, please cite our paper.
```sh
@InProceedings{Miao_2023_ICCV,
    author    = {Miao, Juzheng and Chen, Cheng and Liu, Furui and Wei, Hao and Heng, Pheng-Ann},
    title     = {CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21426-21437}
}
```

## Note
- Please feel free to contact us or open new issues if you encounter any problem when using our code.
- Contact: Juzheng Miao ([jzmiao22@cse.cuhk.edu.hk]())
