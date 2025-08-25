import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.track_predict import predict_tracks

def process_single_folder(model, folder_path, output_path, device, dtype):
    """处理单个文件夹中的所有图片"""
    # 获取所有PNG图片
    image_path_list = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if len(image_path_list) == 0:
        print(f"警告：在{folder_path}中未找到PNG图片")
        return
    
    # 创建输出文件夹
    folder_name = os.path.basename(folder_path)
    folder_output_path = os.path.join(output_path, folder_name)
    os.makedirs(folder_output_path, exist_ok=True)
    
    # 创建子文件夹
    depth_dir = os.path.join(folder_output_path, "depth")
    point_dir = os.path.join(folder_output_path, "point_map")
    feature_dir = os.path.join(folder_output_path, "features")
    confidence_dir = os.path.join(folder_output_path, "confidence")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(point_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(confidence_dir, exist_ok=True)

    # 加载和预处理图片
    vggt_resolution = 518  # VGGT固定分辨率
    images, original_coords = load_and_preprocess_images_square(image_path_list, vggt_resolution)
    images = images.to(device)
    
    # VGGT前向传播
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # 添加batch维度
            images = images[None]
            # 特征提取和聚合
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            
            # 预测相机参数
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # 获取外参和内参矩阵
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # 预测深度图和置信度
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            # 计算3D点云
            points_3d = unproject_depth_map_to_point_map(
                depth_map.squeeze(0).cpu().numpy(),
                extrinsic.squeeze(0).cpu().numpy(),
                intrinsic.squeeze(0).cpu().numpy()
            )

            # 获取特征图
            # 这里我们使用aggregated_tokens作为特征
            features = aggregated_tokens_list[-1]  # 使用最后一层的特征

    # 保存结果
    for idx, img_path in enumerate(image_path_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存深度图
        depth = depth_map[0, idx].cpu().numpy()
        np.save(os.path.join(depth_dir, f"{base_name}_depth.npy"), depth)
        
        # 保存点云图
        point_map = points_3d[idx]
        np.save(os.path.join(point_dir, f"{base_name}_points.npy"), point_map)
        
        # 保存特征图
        feat = features[0, idx].cpu().numpy()
        np.save(os.path.join(feature_dir, f"{base_name}_feat.npy"), feat)
        
        # 保存置信度图
        conf = depth_conf[0, idx].cpu().numpy()
        np.save(os.path.join(confidence_dir, f"{base_name}_conf.npy"), conf)

def main():
    # 设置设备和数据类型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"使用数据类型: {dtype}")

    # 加载VGGT模型
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("模型加载完成")

    # 设置数据集路径
    dataset_root = "dataset"  # 相对路径
    output_root = "processed_data"  # 输出目录
    
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    # 处理DTU数据集
    dtu_path = os.path.join(dataset_root, "DTU")
    if os.path.exists(dtu_path):
        print("处理DTU数据集...")
        scan_folders = glob.glob(os.path.join(dtu_path, "scan*"))
        for folder in tqdm(scan_folders):
            process_single_folder(model, folder, os.path.join(output_root, "DTU"), device, dtype)

    # 处理Replica数据集
    replica_path = os.path.join(dataset_root, "Replica")
    if os.path.exists(replica_path):
        print("处理Replica数据集...")
        room_folders = glob.glob(os.path.join(replica_path, "room*"))
        for folder in tqdm(room_folders):
            process_single_folder(model, folder, os.path.join(output_root, "Replica"), device, dtype)

if __name__ == "__main__":
    main()
