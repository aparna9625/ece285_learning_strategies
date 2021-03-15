# Comparison of learning algorithms for COVID-CT classification
This codebase compares two learning strategies Interleaving learning and Small Group Learning. Further, we also explore the effects of Small Group Learning on the COVID-CT dataset and improvement in image classification results.
In order to run this code, please follow folders corresponding to the task of interest. Each folder contains a README.md to walk through the steps for running the code.
IL-darts - Trains a DARTS based Interleaving Learning model on CIFAR10/100
SGL-pc-darts - Trains a PC-DARTS based Small Group Learning model on CIFAR10/100
SGL-covid - Trains a PC-DARTS based Small Group Learning model on COVID-CT data

### Code References
* Partial Channel Connections for Memory-Efficient Differentiable Architecture Search(PC-DARTS) by Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian and Hongkai Xiong. [Code](https://github.com/yuhuixu1993/PC-DARTS), [Paper](https://openreview.net/forum?id=BJlS634tPr)
* Small Group Learning with Application to Neural Architecture Search by Xuefeng Du, Pengtao Xie. (Code provided privately) [Paper](https://arxiv.org/abs/2012.12502)
* COVID-CT-Dataset: A CT Scan Dataset about COVID-19 by Xingyi Yang, Xuehai He, Jinyu Zhao, Yichen Zhang, Shanghang Zhang, Pengtao Xie. [Dataset](https://github.com/UCSD-AI4H/COVID-CT), [Paper](https://arxiv.org/abs/2003.13865)




Work completed in partial fulfillment of course requirements for ECE 285, Deep Generative Models during Winter 2021 at UCSD by Aparna Srinivasan and Shreyas Rajesh
