# SV$^2$CGS: Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion

<!-- Sparse View Gaussian Splatting via Video Completion -->


[//]: # (###  [Project]&#40;https://decayale.github.io/project/SV2CGS/&#41; | [Arxiv]&#40;https://arxiv.org/abs/2312.00451&#41;)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.00451)
[![Project Page](https://img.shields.io/badge/Project-Website-blue?logo=googlechrome&logoColor=blue)](https://decayale.github.io/project/SV2CGS/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)]()
<!-- [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVITA-Group%2FFSGS&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) -->



---------------------------------------------------
<p align="center" >
  <a href="https://youtu.be/7MpSvjz75XQ">
    <img src="resource/youtube_thumb.png" alt="demo" width="85%">
  </a>
</p>


## Clone the Code

```
git clone https://github.com/DecaYale/SV2C_GS.git 

cd SV2C_GS 
git submodule update --init --recursive

```


## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file env.yml
conda activate SV2CGS 
```

### Install 3DGS
```bash
cd FSGS
pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn

# If errors like "fatal error: crypt.h: No such file or directory" are encountered, please install libxcrypt and expose its path to the environment. Then run the commands above again. 
conda install --channel=conda-forge libxcrypt
export CPATH=$PATH_TO_CONDA_ENV/include/ 


```

### Install Diffuser
```bash 
pip install -e ".[torch]"
```
<!-- **CUDA 11.7** is strongly recommended. -->

## Data Preparation
In data preparation step, we download the official datasets and estimate the camera poses with SfM or transform the provided camera parameters to be compatible with 3D Gaussian Splatting.
We may use all the frames for SfM but would train 3D Gaussian Splatting with sparsely sampled views. We do not directly use the reconstructed point clouds from SfM to initialize the 3D Gaussian Splatting. 
<!-- reconstruct the sparse view inputs using SfM using the camera poses provided by datasets. Next, we continue the dense stereo matching under COLMAP with the function `patch_match_stereo` and obtain the fused stereo point cloud from `stereo_fusion`.  --> -->

``` bash
# Step 1: download processed datasets from https://huggingface.co/datasets/decayale/SV2CGS_DATA/tree/main 

# Step 2:  unzip the data to a target position

unzip nerf_llff_data.zip 

unzip DTU.zip 

tar -xvf DL3DV.tar

``` 
<!-- # if you can not install colmap, follow this to build a docker environment
docker run --gpus all -it --name fsgs_colmap --shm-size=32g  -v /home:/home colmap/colmap:latest /bin/bash
apt-get install pip
pip install numpy
python3 tools/colmap_llff.py
```  -->

<!-- We provide both the sparse and dense point cloud after we proprecess them. You may download them [through this link](https://drive.google.com/drive/folders/1lYqZLuowc84Dg1cyb8ey3_Kb-wvPjDHA?usp=sharing). We use dense point cloud during training but you can still try sparse point cloud on your own. -->

## Training
Train on LLFF dataset with 3 views

```bash 
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_llff_train.sh output/llff/
``` 

Train on DTU dataset with 3 views

```bash 
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_dtu_train.sh output/dtu/
``` 

Train on DL3DV dataset with 9 views

```bash 
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_dl3dv_train.sh output/dl3dv/ 9 
``` 


## Rendering and Evaluation
Run the following script to render and evaluate the images.  

Evaluate on LLFF dataset

```bash
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_llff_eval.sh output/llff/
```

Evaluate on DTU dataset
```bash
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_dtu_eval.sh output/dtu/
```

Evaluate on DL3DV dataset
```bash
# modify the variable $dataset_root according to the location where you unzipped the data before running the script
bash bash_scripts/batch_dl3dv_eval.sh output/dl3dv/
```


## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [dust3r](https://github.com/naver/dust3r)
- [GMFlow](https://github.com/haofeixu/gmflow)

## Citation
If you find our work useful for your project, please consider citing the following paper.


```
@misc{xu2025SV2CGS, 
title={Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion}, 
author={Yan Xu and Yixing Wang and Stella X. Yu}, 
year={2025},
eprint={},
archivePrefix={arXiv},
primaryClass={cs.CV} 
}
```
