# Multi-scale Adaptive Task Attention Network for Few-Shot Learning
This code implements the Multi-scale Adaptive Task Attention Network (MATANet).

Our code is based on [CovaMNet](https://github.com/WenbinLee/CovaMNet).

## Citation
If you find our work useful, please consider citing our work using the bibtex:
```
@inproceedings{chen2020multi,
	author  = {Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
	title   = {Multi-scale Adaptive Task Attention Network for Few-Shot Learning},
	booktitle = {ICPR},
	year    = {2022},
}
```

## Prerequisites
* Linux
* Python 3.6
* Pytorch 1.0+
* GPU + CUDA CuDNN
* pillow, torchvision, scipy, numpy

## Datasets
**Dataset download link:**
* [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view)
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [Stanford Dog](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [Stanford Car](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

**Note: You need to manually change the dataset directory.**

## Few-shot Classification
* Train a 5-way 1-shot model based on Conv-128F (on miniImageNet dataset):
```
python MATA_Train.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --way_num 5 --shot_num 1
```
Test model on the test set:
```
python MATA_Test.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --way_num 5 --shot_num 1
./results/MATA_miniImageNet_MATA_5Way_1Shot_K5/model_best.pth.tar --basemodel MATA
```

## Contacts
Please feel free to contact us if you have any problems.

Email: haoxingchen@smail.nju.edu.cn


