# train_udf.py
import argparse
from typing import Tuple
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import yaml

from dataset import ProcessedSceneDataset
from losses import chamfer_loss, depth_loss, feature_alignment_loss, track_loss
from models.udf import FeatureHead, UDF, volume_render_rays


def get_rays(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ray origins and directions for given pixel coordinates."""
    if intrinsic.ndim == 3:
        intrinsic = intrinsic.squeeze(0)
    if extrinsic.ndim == 3:
        extrinsic = extrinsic.squeeze(0)

    device = intrinsic.device
    K_inv = torch.inverse(intrinsic)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam_o = (-R.transpose(0, 1) @ t).to(device)

    ones = torch.ones_like(pixel_x, dtype=intrinsic.dtype, device=device)
    pix = torch.stack([pixel_x, pixel_y, ones], dim=-1).float()
    dirs_cam = (K_inv @ pix.T).T  # (N,3)
    dirs_world = (R.transpose(0, 1) @ dirs_cam.T).T
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
    origins = cam_o.expand_as(dirs_world)
    return origins, dirs_world


def sample_pixels(confidence: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample pixel coordinates proportional to confidence."""
    h, w = confidence.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=confidence.device),
        torch.arange(w, device=confidence.device),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    probs = confidence.reshape(-1)
    probs = probs / probs.sum()
    indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    sel = coords[indices]
    return sel[:, 0], sel[:, 1]


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)
    scaler = GradScaler()  # 混合精度训练

    # 加载数据集
    dataset = ProcessedSceneDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample_item = dataset[0]
    feature_dim = sample_item["features"].shape[-1]

    # 调度损失权重
    def get_loss_weights(step):
        progress = step / args.max_steps
        w_feat = max(0.2, 0.8 * (1 - progress))  # 从0.8逐渐降到0.2
        w_depth = min(1.0, 0.2 + 0.8 * progress)  # 从0.2逐渐升到1.0
        w_pc = min(1.0, 0.2 + 0.8 * progress)     # 从0.2逐渐升到1.0
        return w_feat, w_depth, w_pc

    udf = UDF().to(device)
    feat_head = FeatureHead(feature_dim=args.feature_dim).to(device) if args.feature_dim > 0 else None
    projector = (
        nn.Linear(feature_dim, args.feature_dim).to(device) if feat_head is not None else None
    )

    params = list(udf.parameters())
    if feat_head is not None:
        params += list(feat_head.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)

    for step, batch in enumerate(loader):
        # 获取训练权重
        w_feat, w_depth, w_pc = get_loss_weights(step)
        
        # 移动数据到设备
        depth = batch["depth"].to(device).squeeze(0)
        points = batch["points"].to(device).squeeze(0)
        features = batch["features"].to(device).squeeze(0)
        confidence = batch["confidence"].to(device).squeeze(0)
        intrinsic = batch.get("intrinsic")
        extrinsic = batch.get("extrinsic")
        if intrinsic is None or extrinsic is None:
            raise ValueError("Dataset must provide camera intrinsics and extrinsics for training")
        intrinsic = intrinsic.to(device).squeeze(0)
        extrinsic = extrinsic.to(device).squeeze(0)

        # 混合精度训练
        with autocast():
            # 采样高置信度像素
            px, py = sample_pixels(confidence, args.rays_per_step)
            gt_depth = depth[py, px]
            gt_conf = confidence[py, px]
            gt_feat = features[py, px]  # (N, C)
            
            # coarse-to-fine采样
            rays_o, rays_d = get_rays(intrinsic, extrinsic, px.float(), py.float())
            
            # 第一阶段：粗采样
            pred_depth_coarse, pred_feat_coarse = volume_render_rays(
                udf, feat_head, rays_o, rays_d, args.near, args.far, 
                num_samples=args.coarse_samples
            )
            
            # 第二阶段：细采样(在高density区域周围)
            with torch.no_grad():
                t_coarse = pred_depth_coarse[..., None]
                t_fine = torch.linspace(
                    t_coarse - args.fine_range, t_coarse + args.fine_range,
                    args.fine_samples, device=device
                )
            
            pred_depth, pred_feat = volume_render_rays(
                udf, feat_head, rays_o, rays_d, args.near, args.far,
                num_samples=args.fine_samples, t_vals=t_fine
            )
            
            # 计算各种损失
            loss = 0.0
            
            # 1. 深度损失
            loss_d = depth_loss(pred_depth, gt_depth, gt_conf)
            loss = loss + w_depth * loss_d
            
            # 2. 点云Chamfer损失
            pred_points = rays_o + rays_d * pred_depth[..., None]
            loss_pc = chamfer_loss(pred_points, points)
            loss = loss + w_pc * loss_pc
            
            # 3. 特征对齐损失
            if feat_head is not None:
                proj_feat = projector(gt_feat)
                loss_feat = feature_alignment_loss(pred_feat, proj_feat, gt_conf)
                loss_contrast = contrastive_feature_loss(pred_feat, proj_feat)
                loss = loss + w_feat * (loss_feat + 0.1 * loss_contrast)
            
            # 4. 正则化项
            loss_tv = tv_regularization(udf, pred_points)
            loss_eik = eikonal_loss(udf, pred_points)
            loss_geo = geometric_regularization(pred_points)
            loss = loss + args.w_reg * (loss_tv + loss_eik + loss_geo)

        writer.add_scalar("loss/total", loss.item(), step)
        if args.w_depth > 0:
            writer.add_scalar("loss/depth", loss_d.item(), step)
        if args.w_pc > 0:
            writer.add_scalar("loss/pc", loss_pc.item(), step)
        if feat_head is not None and args.w_feat > 0:
            writer.add_scalar("loss/feat", loss_feat.item(), step)
        if "tracks" in batch and args.w_track > 0:
            writer.add_scalar("loss/track", loss_tr.item(), step)

        if step % 100 == 0:
            writer.add_image("depth/gt", (depth / depth.max()).unsqueeze(0), step)
            writer.add_mesh("pointcloud/pred", vertices=pred_points.unsqueeze(0), global_step=step)
            writer.add_mesh("pointcloud/gt", vertices=gt_point.unsqueeze(0), global_step=step)

        # 混合精度反向传播
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if step % 10 == 0:
            print(f"step {step}: loss={loss.item():.4f} depth={loss_d.item():.4f} "
                  f"pc={loss_pc.item():.4f} feat={loss_feat.item():.4f}")
            
        # 学习率衰减
        if args.lr_decay:
            current_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * step / args.max_steps))
            for param_group in optim.param_groups:
                param_group["lr"] = current_lr
        
        if step + 1 >= args.max_steps:
            break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UDF model from VGGT processed data")
    parser.add_argument("--config", type=str, help="Path to YAML config file", default=None)
    parser.add_argument("--data-root", type=str, default="processed_data")
    parser.add_argument("--rays-per-step", type=int, default=1024)
    parser.add_argument("--coarse-samples", type=int, default=64, help="Number of coarse samples per ray")
    parser.add_argument("--fine-samples", type=int, default=64, help="Number of fine samples per ray")
    parser.add_argument("--fine-range", type=float, default=0.1, help="Range around coarse samples for fine sampling")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-pc", type=float, default=1.0)
    parser.add_argument("--w-feat", type=float, default=0.5)
    parser.add_argument("--w-track", type=float, default=0.1)
    parser.add_argument("--w-reg", type=float, default=0.1, help="Weight for regularization losses")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--log-dir", type=str, default="runs/udf")
    parser.add_argument("--lr-decay", action="store_true", help="Enable cosine learning rate decay")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    train(args)