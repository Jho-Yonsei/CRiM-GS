<p align="center">
<h1 align="center">
  <img src="./assets/cream_rect.png" align="top" width="38" height="38" />&nbsp;<a href="https://Jho-Yonsei.github.io/CRiM-Gaussian/">CRiM-GS</a>: Continuous Rigid Motion-Aware
  <br />Gaussian Splatting from Motion Blur Images
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

| 3D-GS    | BAGS | CRiM-GS |
| :------: | :------: | :------:
| <img width="100%" src="./assets/gif/factory_3dgs.gif">  |  <img width="100%" src="./assets/gif/factory_bags.gif">   |<img width="100%" src="./assets/gif/factory_crimgs.gif">|

\
We propose continuous rigid motion-aware gaussian splatting (CRiM-GS) to reconstruct accurate 3D scene from blurry images. Considering the actual camera motion blurring process, we predict the continuous movement of the camera based on neural ordinary differential equations (ODEs). Specifically, we introduce continuous rigid body transformations to model the camera motion with proper regularization and a continuous deformable 3D transformation to adapt the rigid body transformation to real-world problems by ensuring a higher degree of freedom. By revisiting fundamental camera theory and employing advanced neural network training techniques, we achieve accurate modeling of continuous camera trajectories.

## Main Framework
<img width="100%" src="./assets/framework.png">

### BibTex
```
@article{lee2024crim,
  title={CRiM-GS: Continuous Rigid Motion-Aware Gaussian Splatting from Motion Blur Images},
  author={Lee, Junghe and Kim, Donghyeong and Lee, Dogyoon and Cho, Suhwan and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2407.03923},
  year={2024}
}
```


### Setup and Training for CRiM-GS
The code will be released soon!
