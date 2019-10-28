# Video-based Person Re-identification
Source code for the ICCV-2019 paper "Co-segmentation Inspired Attention Networks for Video-based Person Re-identification". Our paper can be found <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Subramaniam_Co-Segmentation_Inspired_Attention_Networks_for_Video-Based_Person_Re-Identification_ICCV_2019_paper.pdf" target="_blank">here</a>.

## Introduction

Our work attempts to tackle some of the challenges in Video-based Person re-identification (Re-ID) such as Background clutter, Misalignment error and partial occlusion by means of an co-segmentation inspired approach. The intention is to attend to the task-dependent common portions of the images (i.e., video frames of a person) that may aid the network in better focusing on most relevant features. This repository contains code for Co-segmentation inspired Re-ID architecture, “Co-segmentation Activation Module (COSAM)".
Co-segmentation masks are “Interpretable” and helps to understand how and where the network attends to when creating a description about the person.

### Credits

The source code is built upon the github repositories <a href="https://github.com/jiyanggao/Video-Person-ReID" target="_blank">Video-Person-ReID</a> (from <a href="https://github.com/jiyanggao" target="_blank">jiyanggao</a>) and <a href="https://github.com/KaiyangZhou/deep-person-reid" target="_blank">deep-person-reid</a> (from <a href="https://github.com/KaiyangZhou" target="_blank">KaiyangZhou</a>). Mainly, the data-loading, data-sampling and training part are borrowed from their repository. The strong baseline performances are based on the models from the codebase <a href="https://github.com/jiyanggao/Video-Person-ReID" target="_blank">Video-Person-ReID</a>. Check out their papers <a href="https://arxiv.org/abs/1805.02104" target="_blank">Revisiting Temporal Modeling for Video-based Person ReID (Gao et al.,)</a>, <a href="https://arxiv.org/abs/1905.00953" target="_blank">OSNet (Zhou et al., ICCV 2019)</a>.

We would like to thank <a href="https://github.com/jiyanggao" target="_blank">jiyanggao</a> and <a href="https://github.com/KaiyangZhou" target="_blank">KaiyangZhou</a> for their generous contribution to release the code to the community.

## Datasets

Dataset preparation instructions can be found in the repositories <a href="https://github.com/jiyanggao/Video-Person-ReID" target="_blank">Video-Person-ReID</a> and <a href="https://github.com/KaiyangZhou/deep-person-reid" target="_blank">deep-person-reid</a>. For completeness, I have compiled the dataset instructions <a href="./DATASETS.md" target="_blank">here</a>.

## Training

Will be updated soon.

## Testing

Will be updated soon.