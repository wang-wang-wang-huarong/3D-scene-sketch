import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def process_scene(model, scene_path, output_dir, device):
    """
    处理单个场景文件夹中的所有图片
    Args:
        model: 加载的VGGT模型
        scene_path: 场景文件夹路径 (例如 dataset/DTU/scan105)
        output_dir: 输出目录
        device: 计算设备 (cuda/cpu)
    """
    print(f"Processing scene: {scene_path}")
    
    # 获取场景名称 (例如 scan105)
    scene_name = os.path.basename(scene_path)
    
    # 创建输出目录
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 获取所有图片路径
    image_paths = sorted([str(p) for p in Path(scene_path).glob("*.png")])
    if len(image_paths) == 0:
        print(f"No images found in {scene_path}")
        return
        
    print(f"Found {len(image_paths)} images")

    # 加载和预处理图片
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # 运行模型推理
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            predictions = model(images)
            
    # 获取相机参数
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 计算深度图对应的3D点
    # 遍历每张图片的深度图并处理
    depth_maps = predictions["depth"]  # 应该是 [N, H, W, 1] 或 [N, H, W]
    world_points_list = []
    
    for i in range(len(depth_maps)):
        # 获取单张图片的深度图，确保维度正确 (H, W)
        depth_map = depth_maps[i]
        if depth_map.ndim == 3 and depth_map.shape[-1] == 1:  # [H, W, 1]
            depth_map = depth_map.squeeze(-1)  # 移除最后一个维度变成 [H, W]
            
        # 使用对应的相机参数
        cur_extrinsic = predictions["extrinsic"][i] if predictions["extrinsic"].ndim > 2 else predictions["extrinsic"]
        cur_intrinsic = predictions["intrinsic"][i] if predictions["intrinsic"].ndim > 2 else predictions["intrinsic"]
        
        # 计算当前图片的3D点云
        world_points = unproject_depth_map_to_point_map(depth_map, cur_extrinsic, cur_intrinsic)
        world_points_list.append(world_points)
    
    # 将所有点云合并为一个数组
    predictions["world_points_from_depth"] = np.stack(world_points_list, axis=0)

    # 将所有张量转换为numpy数组并保存
    save_dict = {}
    for key in ["depth", "depth_conf", "world_points", "world_points_conf", 
                "extrinsic", "intrinsic", "world_points_from_depth", 
                "features"]:  # 添加features到保存列表
        if key in predictions:  # 检查键是否存在
            if isinstance(predictions[key], torch.Tensor):
                save_dict[key] = predictions[key].cpu().numpy()
            else:
                save_dict[key] = predictions[key]
    
    # 保存特征
    output_path = os.path.join(scene_output_dir, "features.npz")
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved features to {output_path}")
    
    return output_path

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
        
    # 加载模型
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = model.to(device)
    model.eval()

    # 设置数据集和输出目录
    dataset_dir = "dataset"
    output_base_dir = "processed_features"
    os.makedirs(output_base_dir, exist_ok=True)

    # 处理DTU数据集
    dtu_dir = os.path.join(dataset_dir, "DTU")
    if os.path.exists(dtu_dir):
        for scan_dir in tqdm(sorted(os.listdir(dtu_dir)), desc="Processing DTU scans"):
            scan_path = os.path.join(dtu_dir, scan_dir)
            if os.path.isdir(scan_path):
                process_scene(model, scan_path, os.path.join(output_base_dir, "DTU"), device)

    # 处理Replica数据集
    replica_dir = os.path.join(dataset_dir, "Replica")
    if os.path.exists(replica_dir):
        for room_dir in tqdm(sorted(os.listdir(replica_dir)), desc="Processing Replica rooms"):
            room_path = os.path.join(replica_dir, room_dir)
            if os.path.isdir(room_path):
                process_scene(model, room_path, os.path.join(output_base_dir, "Replica"), device)

if __name__ == "__main__":
    main()