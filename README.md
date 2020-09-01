### Self-supervised Representation Learning from Videos for Facial Action Unit Detection, CVPR 2019 (oral)

We propose a Twin-Cycle Autoencoder (TCAE) that self-supervisedly learns two embeddings to encode the movements of **Facial Actions** and **Head Motions**.

![](img/TCAE_framework.jpg)
*Given a source and target facial images, TCAE is tasked to change the AUs or head poses of the source frame to those of the target frame by predicting the AU-related and pose-related movements, respectively.*

The generated AU-changed and pose-changed faces are shown as below:
![](img/2-cropped.jpg)

*Please refer to the [original paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Self-Supervised_Representation_Learning_From_Videos_for_Facial_Action_Unit_Detection_CVPR_2019_paper.pdf) and [supplementary file](http://openaccess.thecvf.com/content_CVPR_2019/supplemental/Li_Self-Supervised_Representation_Learning_CVPR_2019_supplemental.pdf) for more examples.*

### Prerequisites

- Python 2.x
- Pytorch 0.4.1

### Training yourself

- Download the training dataset: [Voxceleb1/2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- Extract the frames at 1fps, then detect & align the faces, *organize the face directories in format: data/id09238/VqEJCd7pbgQ/0_folder/img_0_001.jpg*
- Split the face images for training/validation/testing by **dataset_split.py**
- Train TCAE by **self_supervised_train_TCAE.py**

*The learned AU embedding from TCAE can be used for both **AU detection** and **facial image retrieval**.*

### Pretrained model released !

- Download pretrained model and script at: [onedrive](https://1drv.ms/u/s!AjMhxexGSrsZmg2TxGQzVgfLwGQu?e=OphhYc)

### Frequently asked questions..

- If you adopt the released model, you should align the facial images in AU datasets by Seetaface. [link](https://github.com/seetafaceengine/SeetaFace2)
- If you train the model yourself, the facial images in voxceleb1/2 should be detected & aligned & cropped, as done in face recognition task.

### If you use this code in your paper, please cite the following:
```
@InProceedings{Li_2019_CVPR,
author = {Li, Yong and Zeng, Jiabei and Shan, Shiguang and Chen, Xilin},
title = {Self-Supervised Representation Learning From Videos for Facial Action Unit Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
