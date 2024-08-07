<p align="center">
<h1 align="center">
  <a href="https://Jho-Yonsei.github.io/CRiM-Gaussian/">CRiM-GS</a>: Continuous Rigid Motion-Aware
  <br />Gaussian Splatting from Motion Blur Images
  <br /><img width="40%" src="./assets/crim-gs.gif">
</h1>
  <p align="center">
    <a href="https://Jho-Yonsei.github.io/">Jungho Lee</a>
    ·
    <a href="https://scholar.google.com/citations?user=BaFYtwgAAAAJ&hl=ko">Donghyeong Kim</a>
    ·
    <a href="https://dogyoonlee.github.io/">Dogyoon Lee</a>
    ·
    <a href="https://suhwan-cho.github.io/">Suhwan Cho</a>
    ·
    <a href="http://mvp.yonsei.ac.kr/">Sangyoun Lee</a>
  </p>
  <p align="center">
	<a href="https://Jho-Yonsei.github.io/CRiM-Gaussian"><img src="https://img.shields.io/badge/CRiM--GS-ProjectPage-white.svg"></a>
     <a href="http://arxiv.org/abs/2407.03923"><img src="https://img.shields.io/badge/CRiM--GS-arXiv-red.svg"></a>
    <a href="https://"><img src="https://img.shields.io/badge/CRiM--GS-Video-yellow.svg"></a>
</p>
  <div align="center"></div>
</p>
<br/>
<br>

| 3D-GS    | BAGS | CRiM-GS (Ours) |
| :------: | :------: | :------:
| <img width="100%" src="./assets/gif/factory_3dgs.gif">  |  <img width="100%" src="./assets/gif/factory_bags.gif">   |<img width="100%" src="./assets/gif/factory_crimgs.gif">|

## Main Framework
<img width="100%" src="./assets/framework.png">

We propose continuous rigid motion-aware gaussian splatting (CRiM-GS) to reconstruct accurate 3D scene from blurry images. Considering the actual camera motion blurring process, we predict the continuous movement of the camera based on neural ordinary differential equations (ODEs). Specifically, we introduce continuous rigid body transformations to model the camera motion with proper regularization and a continuous deformable 3D transformation to adapt the rigid body transformation to real-world problems by ensuring a higher degree of freedom. By revisiting fundamental camera theory and employing advanced neural network training techniques, we achieve accurate modeling of continuous camera trajectories.

## Installation
Clone the repository and create an anaconda environment using.

```
git clone https://github.com/Jho-Yonsei/CRiM-GS.git
cd CRiM-GS

conda create -y -n crimgs python=3.8
conda activate crimgs

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```

Please note that the ```diff-gaussian-rasterization``` we provide is not completely same with the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), as our CRiM kernel optimization requires gradient computation of camera poses. We referred [iComMa](https://github.com/YuanSun-XJTU/iComMa) repository and revised some parts of the cuda-coded backward computation of rasterization.

## Datasets
We have run the COLMAP on the ```synthetic``` scenes, as the synthetic dataset of the original [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF) does not have COLMAP results. We also have run the COLMAP for the ```heron``` and the ```girl``` scenes of ```real-world``` dataset, since their COLMAP results and the image indices are not correctly aligned. You can find the datasets with the COLMAP results on [Our Google Drive](https://drive.google.com/file/d/1P-Z3rp0unw8miQOahbTHdhy0EYVHakZh/view?usp=sharing). The ```real_camera_motion_blur``` and ```synthetic_camera_motion_blur``` directory should be placed in the subdirectory of ```CRiM-GS/dataset/```.

## Training and Evaluation
To reproduce the performance of the paper, then you should run following example commands:
```
# Only for training
python3 ./scripts/run_deblur.py --gpu {gpu} --scene {scene} --train

# For training and rendering
python3 ./scripts/run_deblur.py --gpu {gpu} --scene {scene} --train --render

# For rendering and evaluation
python3 ./scripts/run_deblur.py --gpu {gpu} --scene {scene} --render --metrics

# For pose optimization after training [ Refer our Appendix ]
python3 ./scripts/run_deblur.py --gpu {gpu} --scene {scene} --render --pose_optimize
```

You don't have to specify if the given scene is in ```synthetic``` or ```real-world``` datasets.

## Hyperparameter Setting
To get the ablative results or set other hyperparameters, then run the following example commands:
```
python3 train.py -s ./dataset/synthetic_camera_motion_blur/blurfactory/
                 -m ./work_dir/synthetic_camera_motion_blur/factory/
                 -r 1
                 --llffhold 8
                 --eval
                 --port 6009
                 --num_warp {OPTION}
                 --start_warp {OPTION}
                 --start_pixel_weight {OPTION}
```
Please refer to the ```CRiMParams``` class of ```arguments/__init__.py``` file if you want to see more hyperparameters.

## Online Viewer
If you want to render the trained model in online, use the online viewer of [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting-demo). After run the following command, put the generated ```.ply``` file on the online viewer page.
```
python3 create_fused_ply.py -m {model_dir}/{scene} --output_ply ./fused/{scene}_fused.ply
```

## Pretrained Models
You can find the pretrained Gaussian models for every scene from [Our Google Drive](). Please note that the performances are not completely same with those in the paper because of our code refactoring.

## Acknowledgements
Our repository is built upon [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). The gradient computation of camera poses and pixel-wise weight are influenced by [iComMa](https://github.com/YuanSun-XJTU/iComMa) and [BAGS](https://github.com/snldmt/BAGS), respectively. We thank to all the authors for their awesome works.

## BibTex
```
@article{lee2024crim,
  title={CRiM-GS: Continuous Rigid Motion-Aware Gaussian Splatting from Motion Blur Images},
  author={Lee, Junghe and Kim, Donghyeong and Lee, Dogyoon and Cho, Suhwan and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2407.03923},
  year={2024}
}
```
