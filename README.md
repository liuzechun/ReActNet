# ReActNet

This is the pytorch implementation of our paper ["ReActNet: Towards Precise Binary NeuralNetwork with Generalized Activation Functions"](https://arxiv.org/abs/2003.03488), published in ECCV 2020. 

<div align=center>
<img width=60% src="https://github.com/liuzechun0216/images/blob/master/reactnet_github.jpg"/>
</div>

In this paper, we propose to generalize the traditional Sign and PReLU functions to RSign  and  RPReLU, which enable explicit learning of the distribution reshape and shift at near-zero extra cost. By adding simple learnable bias, ReActNet achieves 69.4% top-1 accuracy on Imagenet dataset with both weights and activations being binary, a near ResNet-level accuracy.

## Citation

If you find our code useful for your research, please consider please consider citing:

    @inproceedings{liu2020reactnet,
      title={ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions},
      author={Liu, Zechun and Shen, Zhiqiang and Savvides, Marios and Cheng, Kwang-Ting},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2020}
    }

## Run

### 1. Requirements:
* python3, pytorch 1.4.0, torchvision 0.5.0
    
### 2. Data:
* Download ImageNet dataset

### 3. Steps to run:
(1) Step1:  binarizing activations
* Change directory to `./resnet/1_step1/` or `./mobilenet/1_step1/`
* run `bash run.sh`

(2) Step2:  binarizing weights + activations
* Change directory to `./resnet/2_step2/` or `./mobilenet/2_step2/`
* run `bash run.sh`
       

## Models

| Methods | Top1-Acc | FLOPs | Trained Model |
| --- | --- | --- | --- | 
| XNOR-Net | 51.2% | 1.67 x 10^8 | - |
| Bi-Real Net| 56.4% | 1.63 x 10^8 | - | 
| Real-to-Binary| 65.4% | 1.83 x 10^8 | - |
| ReActNet (Bi-Real based) | 65.5% | 1.63 x 10^8 | [Model-ReAct-ResNet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/Eci3hQSA2gpHgiqkQDIcBaIByZk6zwMFGoODp9vjr-6eeA?e=oo12qg) |
| ReActNet-A | 69.5% | 0.87 x 10^8 | [Model-ReAct-MobileNet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EdaAeiqveCtPt3ElnaY4JJgBsaScj5snGkxIuJhuIPd25A?e=sqg1b6) |

## Contact

Zechun Liu, HKUST (zliubq at connect.ust.hk)

Zhiqiang Shen, CMU (zhiqians at andrew.cmu.edu) 
