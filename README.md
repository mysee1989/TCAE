### Self-supervised Representation Learning from Videos for Facial Action Unit Detection, CVPR 2019 (oral)

We propose a Twin-Cycle Autoencoder (TCAE) that self-supervisedly learns two embeddings to encode the movements of AUs and head motions.
<br />Given a source and target facial images, TCAE is tasked to change the AUs or head poses of the source frame to those of the target frame by predicting the AU-related and pose-related movements, respectively. 

![image](https://github.com/mysee1989/TCAE/blob/master/img/2-cropped.jpg)

<br />After training, the learned encoder can be used for AU detection. The extracted AU embedding from the encoder can be used for both AU detection and facial image retrieval.

### If you use this code in your paper, please cite the following:
```
@inproceedings{li2019self,
  title={Self-supervised Representation Learning from Videos for Facial Action Unit Detection},
  author={Li, Yong and Zeng, Jiabei and Shan, Shiguang and Chen, Xilin},
  booktitle={CVPR},
  year={2019}
}
```
