# SGL

Code accompanying the paper  
***Small Group Learning, with Application to Neural Architecture Search*** [paper]()  
<!-- Xiangning Chen, Ruochen Wang, Minhao Cheng, Xiaocheng Tang, Cho-Jui Hsieh -->

This code is based on the implementation of [P-DARTS](https://github.com/chenxin061/pdarts), [DARTS](https://github.com/quark0/darts) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Architecture Search

<!-- **Search on NAS-Bench-201 Space: (3 datasets to choose from)**

* Data preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/NAS-Bench-201).

* ```cd 201-space && python train_search_progressive.py``` -->


**Composing SGL with PC-DARTS:**

```
CIFAR-10: cd SGL-pc-darts && python train_search_coop_pretrain.py --weight_lambda 1 \\
--set cifar10 --gpu 0 --batch_size 50 --save xxx
```

```
CIFAR-100: cd SGL-pc-darts && python train_search_coop_pretrain.py --weight_lambda 1 \\
--set cifar100 --gpu 0 --batch_size 50 --save xxx
```

* Data preparation: Please first sample 10% and 2.5% images for each class as the training and validation set, which is done by pcdarts-LCT/sample_images.py.

```
ImageNet: cd SGL-pc-darts && python train_search_coop_pretrain.py --save xxx --tmp_data_dir xxx \\
--weight_lambda 1
```

where you can change the value of lambda.

## Architecture Evaluation

```
ImageNet: cd SGL-pdarts && python train_imagenet.py --auxiliary --arch xxx
```

**Composing LCT with PC-DARTS:**

```
ImageNet: cd SGL-pc-darts && python train.py --cutout --auxiliary \\
--set cifar10/100 --arch xxx \\
--seed 3 --save xxx
```

```
ImageNet: cd SGL-pc-darts && python train_imagenet.py --note xxx --auxiliary --arch xxx
```


## Ablation Study (Search)

You can change the --pretrain_epochs to 0 for DARTS, P-DARTS OR PC-DARTS for ablation study 1 and change the training objective to use solely pseudo labels to search for ablation study 2.


The evaluation is the same as the above.
