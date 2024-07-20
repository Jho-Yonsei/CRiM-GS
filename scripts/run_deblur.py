import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
from argparse import ArgumentParser


def reconstruct_scene(gpu: int = 0,
                      train: bool = True,
                      render: bool = True,
                      pose_optimize: bool = False,
                      metrics: bool = True,
                      scene: str = 'factory', 
                      start_warp: int = 1000, 
                      start_pixel_weight: int = 3000, 
                      gamma_correction: bool = False, 
                      skip_train: bool = True, 
                      skip_test: bool = False, 
                      skip_video: bool = False) -> bool:
    
    if scene in synthetic_scenes:
        scene_type = "synthetic"
        scene_factor = synthetic_factor
        llffhold = synthetic_dict[scene]
    elif scene in real_scenes:
        scene_type = "real"
        scene_factor = real_factor
        llffhold = real_dict[scene]
    else:
        raise ValueError(f"Unknown scene {scene}")
    
    if train:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s ./dataset/{scene_type}_camera_motion_blur/blur{scene}/ -m ./work_dir/{scene_type}_camera_motion_blur/{scene}/ --eval -r {scene_factor} --llffhold {llffhold} --port {6009+int(gpu)} --kernel_size 0.1"
        if start_warp is not None:
            cmd += f" --start_warp {start_warp}"
        if start_pixel_weight is not None:
            cmd += f" --start_pixel_weight {start_pixel_weight}"
        if gamma_correction:
            cmd += " --gamma_correction"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    if render:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m ./work_dir/{scene_type}_camera_motion_blur/{scene}/ --data_device cpu"
        if skip_train:
            cmd += " --skip_train"
        if skip_test:
            cmd += " --skip_test"
        if skip_video:
            cmd += " --skip_video"
        if gamma_correction:
            cmd += " --gamma_correction"
        if pose_optimize:
            cmd += " --pose_optimize"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    if metrics:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m ./work_dir/{scene_type}_camera_motion_blur/{scene}/ -r {scene_factor}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Script parameters")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--pose_optimize", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--scene", type=str, default='factory', required=True)
    parser.add_argument("--start_warp", type=int, default=1000)
    parser.add_argument("--start_pixel_weight", type=int, default=3000)
    parser.add_argument("--gamma_correction", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    args = parser.parse_args()

    synthetic_scenes = ["factory", "cozy2room", "pool", "tanabata", "wine"]
    synthetic_factor = 1
    synthetic_llffhold = [8] * len(synthetic_scenes)
    synthetic_dict = dict(zip(synthetic_scenes, synthetic_llffhold))

    real_scenes = ["ball", "basket", "buick", "coffee", "decoration", "girl", "heron", "parterre", "puppet", "stair"]
    real_factor = 4
    real_llffhold = [7, 7, 7, 6, 6, 7, 8, 6, 6, 6]
    real_dict = dict(zip(real_scenes, real_llffhold))

    excluded_gpus = set([])

    dry_run = False

    if args.scene == "pool":
        args.start_warp = 3000
        args.start_pixel_weight = 6000
        args.gamma_correction = True

    reconstruct_scene(**vars(args))