## Dependencies and Installation
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
The datasets used in this paper can be downloaded in the following links.  
[[Training Dataset]](https://pan.baidu.com/s/19DtLPftHomCb6_1V2lREtw?pwd=3gbv)
[[Testing Dataset]](https://pan.baidu.com/s/10KzmwC1jncozOGNZ02Hlaw?pwd=sxfd)



## Get Started
Training and testing codes are in the current folder. 

-   The scripts of code is in `./scripts`, please navigate to the script workspace by running `cd ./scriptt`

-   For training, you also need to set the `is_training = 0` in the `main.py` to match your testing dataset. Then, run `bash ./R`

-   For testing, you also need to set the `is_training = 0` in the `main.py` to match your testing dataset. 