# shoulder-c
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classification-of-fracture-and-normal/image-classification-on-fracture-normal)](https://paperswithcode.com/sota/image-classification-on-fracture-normal?p=classification-of-fracture-and-normal)  
PyTorch Implementation for Classification of Fracture/Normal Shoulder Bone X-ray Images
# note
**Published**: 18 March 2021, Uysal, F.; Hardalaç, F.; Peker, O.; Tolunay, T.; Tokgöz, N. Classification of Shoulder X-ray Images with Deep Learning Ensemble Models. Appl. Sci. 2021, 11, 2723. https://doi.org/10.3390/app11062723. IF 2019: 2.474, Index: SCIE, JCR: Q2, EF: 0.02035, AI: 0.351  
# abstract
Fractures occur in the shoulder area, which has a wider range of motion than other joints in the body, for various reasons. To diagnose these fractures, data gathered from X-radiation (X-ray), magnetic resonance imaging (MRI), or computed tomography (CT) are used. This study aims to help physicians by classifying shoulder images taken from X-ray devices as fracture/non-fracture with artificial intelligence. For this purpose, the performances of 26 deep learning-based pre-trained models in the detection of shoulder fractures were evaluated on the musculoskeletal radiographs (MURA) dataset, and two ensemble learning models (EL1 and EL2) were developed. The pre-trained models used are ResNet, ResNeXt, DenseNet, VGG, Inception, MobileNet, and their spinal fully connected (Spinal FC) versions. In the EL1 and EL2 models developed using pre-trained models with the best performance, test accuracy was 0.8455, 0.8472, Cohen’s kappa was 0.6907, 0.6942 and the area that was related with fracture class under the receiver operating characteristic (ROC) curve (AUC) was 0.8862, 0.8695. As a result of 28 different classifications in total, the highest test accuracy and Cohen’s kappa values were obtained in the EL2 model, and the highest AUC value was obtained in the EL1 model.  

**Keywords**: biomedical image classification; bone fractures; deep learning; ensemble learning; shoulder; transfer learning; X-ray
# paper links
**Paper**: https://www.mdpi.com/2076-3417/11/6/2723  

**Preprint**: http://arxiv.org/abs/2102.00515  

**Papers With Code**: https://paperswithcode.com/paper/classification-of-fracture-and-normal  

**GitHub**: https://github.com/fatihuysal88/shoulder-c  
# citation
please cite the paper if you benefit from our paper:  
```
@article{Uysal_2021, title={Classification of Shoulder X-ray Images with Deep Learning Ensemble Models}, volume={11}, ISSN={2076-3417}, url={http://dx.doi.org/10.3390/app11062723}, DOI={10.3390/app11062723}, number={6}, journal={Applied Sciences}, publisher={MDPI AG}, author={Uysal, Fatih and Hardalaç, Fırat and Peker, Ozan and Tolunay, Tolga and Tokgöz, Nil}, year={2021}, month={Mar}, pages={2723}}
```
# authors
* **Fatih UYSAL** - [PhD Candidate and Research Assistant at Gazi University, Republic of Turkey.](https://orcid.org/0000-0002-1731-2647)
* **Fırat HARDALAÇ** - [Prof. at Gazi University, Republic of Turkey.](https://orcid.org/0000-0003-1358-0756)
* **Ozan PEKER** - [MSc student at Gazi University, Republic of Turkey.](https://orcid.org/0000-0003-2258-1531)
* **Tolga TOLUNAY** - [Assoc. Prof. at Gazi University, Republic of Turkey.](https://orcid.org/0000-0003-1998-3695)
* **Nil TOKGÖZ** - [Prof. at Gazi University, Republic of Turkey.](https://orcid.org/0000-0003-2812-1528)
# proposed classification models
![models](https://github.com/fatihuysal88/shoulder-c/blob/main/docs/figs/proposed%20classification%20models.png)
# models performance
<img src="https://github.com/fatihuysal88/shoulder-c/blob/main/docs/figs/Cohen%20Kappa%20Score%20Comparison.png" width="718" height="250">
<img src="https://github.com/fatihuysal88/shoulder-c/blob/main/docs/figs/Test%20Accuracy%20Comparison.png" width="718" height="250">
<img src="https://github.com/fatihuysal88/shoulder-c/blob/main/docs/figs/AUC%20Score%20Comparison.png" width="718" height="250">  
