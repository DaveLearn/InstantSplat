# This code was based on code from https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py
# and is based on the algorim described in the paper "Massively Parallel Multiview Stereopsis by Surface Normal Diffusion"

from typing import List, Tuple
import torch
import torch
import torch.nn.functional as F
import torch.nn.parallel

import torch.nn.functional as F

from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import fov2focal


# read intrinsics and extrinsics
def get_camera_parameters(cam: Camera, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]: # (intrinsics, extrinsics)
    
    assert isinstance(cam, Camera), "cam must be an instance of Camera, but was {}".format(type(cam))

    extrinsics = torch.cat((torch.tensor(cam.R, dtype=torch.float32, device=device), 
                           torch.tensor(cam.T.reshape(3, 1), dtype=torch.float32, device=device)), dim=1)
    extrinsics = torch.cat((extrinsics, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)), dim=0)

    focal_length_y = fov2focal(cam.FoVy, cam.image_height)
    focal_length_x = fov2focal(cam.FoVx, cam.image_width)

    intrinsics = torch.tensor([[focal_length_x, 0, cam.image_width/2],
                               [0, focal_length_y, cam.image_height/2],
                               [0, 0, 1]], dtype=torch.float32, device=device)
    return intrinsics, extrinsics



def reproject_with_depth(depth_ref: torch.Tensor, intrinsics_ref: torch.Tensor, extrinsics_ref: torch.Tensor,
                         depth_src: torch.Tensor, intrinsics_src: torch.Tensor, extrinsics_src: torch.Tensor):
    """
    Reprojects a depth map from the reference view to the source view and back to the reference view.

    Args:
        depth_ref (torch.Tensor): Depth map of the reference view (shape: [H, W]).
        intrinsics_ref (torch.Tensor): Intrinsic matrix of the reference camera (shape: [3, 3]).
        extrinsics_ref (torch.Tensor): Extrinsic matrix of the reference camera (shape: [4, 4]).
        depth_src (torch.Tensor): Depth map of the source view (shape: [H, W]).
        intrinsics_src (torch.Tensor): Intrinsic matrix of the source camera (shape: [3, 3]).
        extrinsics_src (torch.Tensor): Extrinsic matrix of the source camera (shape: [4, 4]).

    Returns:
        depth_reprojected (torch.Tensor): Reprojected depth map in the reference view (shape: [H, W]).
        x_reprojected (torch.Tensor): Reprojected x-coordinates in the reference view (shape: [H, W]).
        y_reprojected (torch.Tensor): Reprojected y-coordinates in the reference view (shape: [H, W]).
        x_src (torch.Tensor): Source view x-coordinates (shape: [H, W]).
        y_src (torch.Tensor): Source view y-coordinates (shape: [H, W]).
    """
       # Validate input shapes
    assert depth_ref.dim() == 2, "depth_ref must be a 2D tensor of shape [H, W]"
    assert depth_src.dim() == 2, "depth_src must be a 2D tensor of shape [H, W]"
    assert intrinsics_ref.shape == (3, 3), "intrinsics_ref must be a 3x3 tensor"
    assert intrinsics_src.shape == (3, 3), "intrinsics_src must be a 3x3 tensor"
    assert extrinsics_ref.shape == (4, 4), "extrinsics_ref must be a 4x4 tensor"
    assert extrinsics_src.shape == (4, 4), "extrinsics_src must be a 4x4 tensor"
    assert depth_ref.shape == depth_src.shape, "depth_ref and depth_src must have the same shape"


    height, width = depth_ref.shape
    device = depth_ref.device

    # Step 1: Project reference pixels to the source view
    # Generate pixel coordinates for the reference view
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width, device=device),
                                  torch.arange(0, height, device=device), indexing='xy')
    x_ref, y_ref = x_ref.reshape(-1), y_ref.reshape(-1)  # Flatten to 1D

    # Convert reference pixel coordinates to 3D points in the reference camera space
    ones = torch.ones_like(x_ref)
    xy1_ref = torch.stack((x_ref, y_ref, ones), dim=0).float()  # Shape: [3, H*W]
    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref), xy1_ref) * depth_ref.reshape(-1)  # Shape: [3, H*W]

    # Transform 3D points from the reference camera space to the source camera space
    xyz1_ref = torch.vstack((xyz_ref, ones))  # Shape: [4, H*W]
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)), xyz1_ref)[:3]  # Shape: [3, H*W]

    # Project 3D points in the source camera space to source pixel coordinates
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # Shape: [3, H*W]
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]  # Perspective division, shape: [2, H*W]

    # Reshape source pixel coordinates to 2D grids
    x_src = xy_src[0].reshape(height, width)  # Shape: [H, W]
    y_src = xy_src[1].reshape(height, width)  # Shape: [H, W]

    # to help debug visually, plot these using scatter, color based on their depth (from K_xyz_src)
    # overlay that on the source depth map
    import matplotlib.pyplot as plt
    plt.imshow(depth_src.cpu().numpy(), cmap='gray')
    plt.scatter(x_src.cpu().numpy(), y_src.cpu().numpy(), c=K_xyz_src[2].cpu().numpy(), cmap='viridis')
    plt.show()


    # Step 2: Reproject the source view points with source view depth estimation
    # Normalize source pixel coordinates to the range [-1, 1] for grid_sample
    x_src_norm = (x_src / (width - 1)) * 2 - 1  # Shape: [H, W]
    y_src_norm = (y_src / (height - 1)) * 2 - 1  # Shape: [H, W]
    grid = torch.stack((x_src_norm, y_src_norm), dim=-1).unsqueeze(0)  # Shape: [1, H, W, 2]

    # Sample the source depth map at the computed pixel coordinates using bilinear interpolation
    sampled_depth_src = F.grid_sample(depth_src.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True)
    sampled_depth_src = sampled_depth_src.squeeze(0).squeeze(0)  # Shape: [H, W]

    # Convert source pixel coordinates back to 3D points in the source camera space
    xy1_src = torch.vstack((xy_src, ones))  # Shape: [3, H*W]
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src), xy1_src) * sampled_depth_src.reshape(-1)  # Shape: [3, H*W]

    # Transform 3D points from the source camera space back to the reference camera space
    xyz1_src = torch.vstack((xyz_src, ones))  # Shape: [4, H*W]
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)), xyz1_src)[:3]  # Shape: [3, H*W]

    # Project the reprojected 3D points to the reference pixel coordinates
    depth_reprojected = xyz_reprojected[2].reshape(height, width)  # Shape: [H, W]
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)  # Shape: [3, H*W]
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]  # Shape: [2, H*W]

    # Reshape reprojected pixel coordinates to 2D grids
    x_reprojected = xy_reprojected[0].reshape(height, width)  # Shape: [H, W]
    y_reprojected = xy_reprojected[1].reshape(height, width)  # Shape: [H, W]

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref: torch.Tensor, intrinsics_ref: torch.Tensor, extrinsics_ref: torch.Tensor, depth_src: torch.Tensor, intrinsics_src: torch.Tensor, extrinsics_src: torch.Tensor, relative_depth_diff_threshold: float = 0.01):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
    x_ref = x_ref.to(depth_ref.device)
    y_ref = y_ref.to(depth_ref.device)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1 (same pixel)
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < relative_depth_diff_threshold ()
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < 1, relative_depth_diff < relative_depth_diff_threshold)
    depth_reprojected[~mask] = 0

    # print number of pixels that dist < 1, and as a percentage
    print("dist < 1:{}".format(dist[dist < 1].shape[0]))
    print("dist < 1:{}%".format(dist[dist < 1].shape[0] / dist.shape[0] * 100))

    # plot with matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(dist.cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.show()

    # print relative and absolute depth diff
    print("relative_depth_diff_mean:{}".format(relative_depth_diff.mean()))
    print("relative_depth_diff_max:{}".format(relative_depth_diff.max()))
    print("depth_diff_mean:{}".format(depth_diff.mean()))
    print("depth_diff_max:{}".format(depth_diff.max()))

    return mask, depth_reprojected, x2d_src, y2d_src


# filter the depths using geometric consistency returns depth validity mask and an averaged depth
def filter_depths(cam_infos: List[CameraInfo], depth_est: List[torch.Tensor], confidences: List[torch.Tensor], 
                 confidence_threshold: float = 0.8, relative_depth_diff_threshold: float = 0.01, required_views: int = 3
                 ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    
    # Add device consistency checks
    assert len(depth_est) > 0 and len(confidences) > 0, "Empty depth_est or confidences lists"
    device = depth_est[0].device
    assert all(d.device == device for d in depth_est), "All depth_est tensors must be on the same device"
    assert all(c.device == device for c in confidences), "All confidence tensors must be on the same device"


    # quick sanity check against ourself, delete later
    ref_intrinsics, ref_extrinsics = get_camera_parameters(cam_infos[0], device)
    ref_depth_est = depth_est[0]
    src_intrinsics, src_extrinsics = get_camera_parameters(cam_infos[1], device)
    src_depth_est = depth_est[1]

    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics, 0.01)
    print("geo_mask_sum:{}".format(geo_mask.float().mean()))
    print("geo_mask_max:{}".format(geo_mask.max()))

    # check that depth_reprojected is the same as ref_depth_est
    # calc distance between depth_reprojected and ref_depth_est
    depth_diff = torch.abs(depth_reprojected - ref_depth_est)
    print("depth_diff_mean:{}".format(depth_diff.mean()))
    print("depth_diff_max:{}".format(depth_diff.max()))
    
    # print the min and max depth of src_depth_est and ref_depth_est
    print("src_depth_est_min:{}".format(src_depth_est.min()))
    print("src_depth_est_max:{}".format(src_depth_est.max()))
    print("ref_depth_est_min:{}".format(ref_depth_est.min()))
    print("ref_depth_est_max:{}".format(ref_depth_est.max()))
    # and the mean
    print("src_depth_est_mean:{}".format(src_depth_est.mean()))
    print("ref_depth_est_mean:{}".format(ref_depth_est.mean()))

    # and same for depth_reprojected
    print("depth_reprojected_min:{}".format(depth_reprojected.min()))
    print("depth_reprojected_max:{}".format(depth_reprojected.max()))
    print("depth_reprojected_mean:{}".format(depth_reprojected.mean()))

    assert torch.allclose(depth_reprojected, ref_depth_est), "depth_reprojected is not the same as ref_depth_est"



    pair_data = [
        (ref_idx, [src_idx for src_idx in range(len(cam_infos)) if src_idx != ref_idx])
        for ref_idx in range(len(cam_infos))
    ]

    depth_validity_masks = []
    depth_averaged = []

    # for each reference view and the corresponding source views
    for ref_view_idx, src_view_idxs in pair_data:
        # load the camera parameters with the correct device
        ref_intrinsics, ref_extrinsics = get_camera_parameters(cam_infos[ref_view_idx], device)
            
        ref_depth_est = depth_est[ref_view_idx]
        confidence = confidences[ref_view_idx]
        photo_mask = confidence > confidence_threshold

        all_srcview_depth_ests = []
        geo_mask_sum = torch.zeros_like(ref_depth_est)
        
        for src_view_idx in src_view_idxs:
            # camera parameters of the source view with the correct device
            src_intrinsics, src_extrinsics = get_camera_parameters(cam_infos[src_view_idx], device)
            
            # Get the source depth estimation
            src_depth_est = depth_est[src_view_idx]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics, relative_depth_diff_threshold)
            geo_mask_sum += geo_mask.int()
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (torch.stack(all_srcview_depth_ests).sum(dim=0) + ref_depth_est) / (geo_mask_sum + 1)
        # at least required_views source views matched
        geo_mask = geo_mask_sum >= required_views
        final_mask = torch.logical_and(photo_mask, geo_mask)
        print("geo_mask_sum:{}".format(geo_mask_sum.float().mean()))
        print("geo_mask_max:{}".format(geo_mask_sum.max()))
        print("processed {},  photo/geo/final-mask:{}/{}/{}".format(cam_infos[ref_view_idx].image_name,
                                                                                    photo_mask.float().mean(),
                                                                                    geo_mask.float().mean(), final_mask.float().mean()))

        depth_validity_masks.append(final_mask)
        depth_averaged.append(depth_est_averaged)


    return depth_validity_masks, depth_averaged