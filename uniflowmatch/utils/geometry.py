"""
Utils for geometric calculations
Adopted from AnyMap (Nikhil Keetha)
Includes functions from DUSt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only)) & GradSLAM (MIT License)
"""

from functools import lru_cache

import einops as ein
import numpy as np
import torch

from uniflowmatch.utils.misc import invalid_to_nans
from scipy.spatial import KDTree


def depthmap_to_camera_frame(depthmap, intrinsics):
    """
    Convert depth image to a pointcloud in camera frame.
    Args:
        - depthmap: HxW torch tensor
        - camera_intrinsics: 3x3 torch tensor
    Returns:
        pointmap in camera frame (HxWx3 tensor), and a mask specifying valid pixels.
    """
    height, width = depthmap.shape
    device = depthmap.device
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Compute 3D point in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).to(device).float(), torch.arange(height).to(device).float(), indexing="xy"
    )
    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = torch.stack((xx, yy, depth_z), dim=-1)

    # Compute mask of valid non-zero depth pixels
    valid_mask = depthmap > 0.0

    return pts3d_cam, valid_mask


def depthmap_to_world_frame(depthmap, intrinsics, camera_pose=None):
    """
    Convert depth image to a pointcloud in world frame.

    Args:
        - depthmap: HxW torch tensor
        - camera_intrinsics: 3x3 torch tensor
        - camera_pose: 4x4 torch tensor

    Returns:
        pointmap in world frame (HxWx3 tensor), and a mask specifying valid pixels.
    """
    pts3d_cam, valid_mask = depthmap_to_camera_frame(depthmap, intrinsics)

    if camera_pose is not None:
        pts3d_cam_homo = torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], dim=-1)
        pts3d_world = ein.einsum(camera_pose, pts3d_cam_homo, "i k, h w k -> h w i")
        pts3d_world = pts3d_world[..., :3]

    return pts3d_world, valid_mask


def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)

    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and Trf.ndim == 3 and pts.ndim == 4:
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)

    return res


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


@lru_cache(maxsize=10)
def get_meshgrid(W, H):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return u, v


@lru_cache(maxsize=10)
def get_meshgrid_torch(W, H, device):
    u, v = torch.meshgrid(torch.arange(W, device=device).float(), torch.arange(H, device=device).float(), indexing="xy")

    uv = torch.stack((u, v), dim=-1)

    return uv


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = get_meshgrid(W, H)

    X_cam = np.zeros((H, W, 3), dtype=np.float32)

    X_cam[..., 0] = (u - cu) * depthmap / fu
    X_cam[..., 1] = (v - cv) * depthmap / fv
    X_cam[..., 2] = depthmap

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask


def z_depthmap_to_norm_depthmap(z_depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - z_depthmap (HxW array)
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = z_depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    rays = np.ones((H, W, 3), dtype=np.float32)

    u, v = get_meshgrid(W, H)

    rays[..., 0] = (u - cu) / fu
    rays[..., 1] = (v - cv) / fv

    ray_norm = np.linalg.norm(rays, axis=-1)

    return z_depthmap * ray_norm


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        # X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        X_world = X_cam @ (R_cam2world.T) + t_cam2world[None, None, :]

    return X_world, valid_mask


def global_points_to_local(pts, camera_pose):
    """
    Args:
        - pts: points in world coordinate
        - camera_pose: camera to world transformation
    """

    world_to_camera = np.linalg.inv(camera_pose)
    R_world2cam = world_to_camera[:3, :3]
    t_world2cam = world_to_camera[:3, 3]

    pts_local = np.einsum("ik, vuk -> vui", R_world2cam, pts) + t_world2cam[None, None, :]

    return pts_local


def project_points_to_pixels(pts_camera, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - pts_camera (HxWx3 array): points in camera coordinates
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pixel coordinates (HxWx2 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = pts_camera.shape[:2]

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    x, y, z = pts_camera[..., 0], pts_camera[..., 1], pts_camera[..., 2]

    uv = np.zeros((H, W, 2), dtype=np.float32)

    uv[..., 0] = fu * x / z + cu
    uv[..., 1] = fv * y / z + cv

    # Mask for valid coordinates
    valid_mask = (
        (z > 0.0) & (uv[..., 0] >= -0.5) & (uv[..., 0] < W - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] < H - 0.5)
    )
    # valid_mask = (z > 0.0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)

    return uv, valid_mask


def project_points_to_pixels_batched(pts_camera, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - pts_camera (BxHxWx3 torch.Tensor): points in camera coordinates
        - camera_intrinsics: a Bx3x3 torch.Tensor
    Returns:
        pixel coordinates (BxHxWx2 torch.Tensor), and a mask (BxHxW) specifying valid pixels.
    """
    camera_intrinsics = camera_intrinsics
    B, H, W, C = pts_camera.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert (camera_intrinsics[..., 0, 1] == 0.0).all()
    assert (camera_intrinsics[..., 1, 0] == 0.0).all()
    if pseudo_focal is None:
        fu = camera_intrinsics[..., 0, 0]
        fv = camera_intrinsics[..., 1, 1]
    else:
        assert pseudo_focal.shape == (B, H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[..., 0, 2]
    cv = camera_intrinsics[..., 1, 2]

    x, y, z = pts_camera[..., 0], pts_camera[..., 1], pts_camera[..., 2]

    uv = torch.zeros((B, H, W, 2), dtype=pts_camera.dtype, device=pts_camera.device)

    uv[..., 0] = fu.view(B, 1, 1) * x / z + cu.view(B, 1, 1)
    uv[..., 1] = fv.view(B, 1, 1) * y / z + cv.view(B, 1, 1)

    # Mask for valid coordinates
    valid_mask = (
        (z > 0.0) & (uv[..., 0] >= -0.5) & (uv[..., 0] < W - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] < H - 0.5)
    )
    # valid_mask = (z > 0.0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)

    return uv, valid_mask


def z_depthmap_to_norm_depthmap_batched(z_depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - z_depthmap (BxHxW array)
        - camera_intrinsics: a Bx3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """

    B, H, W = z_depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert (camera_intrinsics[..., 0, 1] == 0.0).all()
    assert (camera_intrinsics[..., 1, 0] == 0.0).all()
    if pseudo_focal is None:
        fu = camera_intrinsics[..., 0, 0]
        fv = camera_intrinsics[..., 1, 1]
    else:
        assert pseudo_focal.shape == (B, H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[..., 0, 2]
    cv = camera_intrinsics[..., 1, 2]

    rays = torch.ones((B, H, W, 3), dtype=z_depthmap.dtype, device=z_depthmap.device)

    uv = get_meshgrid_torch(W, H, device=z_depthmap.device)

    rays[..., 0] = (uv[..., 0].view(1, H, W) - cu.view(B, 1, 1)) / fu.view(B, 1, 1)
    rays[..., 1] = (uv[..., 1].view(1, H, W) - cv.view(B, 1, 1)) / fv.view(B, 1, 1)

    ray_norm = torch.linalg.norm(rays, axis=-1)

    return z_depthmap * ray_norm


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)

    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values

    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2))
    reciprocal_in_P2 = nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()

    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def rotate_vector_with_quaternion(
    v: torch.Tensor, quat: torch.Tensor, scalar_first: bool = False, skip_norm=False
) -> torch.Tensor:
    """
    Rotate a 3D vector by a quaternion.

    Args:
        v (torch.Tensor): A tensor of shape (..., 3) representing the vectors to rotate.
        quat (torch.Tensor): A tensor of shape (..., 4) representing the quaternions.
                             The last dimension is [w, x, y, z] if scalar_first is True,
                             or [x, y, z, w] if scalar_first is False.
        scalar_first (bool): If True, assumes the quaternion is in the format [w, x, y, z].
                             Otherwise, assumes the format [x, y, z, w].

    Returns:
        torch.Tensor: A tensor of shape (..., 3) representing the rotated vectors.
    """
    if scalar_first:
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    else:
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Normalize the quaternion to ensure a valid rotation
    if not skip_norm:
        norm_quat = torch.sqrt(w**2 + x**2 + y**2 + z**2 + 1e-8)
        w, x, y, z = w / norm_quat, x / norm_quat, y / norm_quat, z / norm_quat

    # Vector part of the quaternion
    q_vec = torch.stack([x, y, z], dim=-1)  # Shape (..., 3)

    # Cross product q_vec x v
    t = 2 * torch.cross(q_vec, v, dim=-1)  # Intermediate vector, shape (..., 3)

    # Ensure proper broadcasting of w
    v_rotated = v + w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)

    return v_rotated


def quaternion_to_rot_matrix(quat: torch.Tensor, scalar_first: bool = False) -> torch.Tensor:
    if scalar_first:
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    else:
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    norm_quat = torch.sqrt(w**2 + x**2 + y**2 + z**2 + 1e-8)
    w, x, y, z = w / norm_quat, x / norm_quat, y / norm_quat, z / norm_quat

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot_matrix_shape = quat.shape[:-1] + (3, 3)
    rot_matrix = torch.empty(rot_matrix_shape, device=quat.device)

    rot_matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    rot_matrix[..., 0, 1] = 2 * (xy - wz)
    rot_matrix[..., 0, 2] = 2 * (xz + wy)

    rot_matrix[..., 1, 0] = 2 * (xy + wz)
    rot_matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    rot_matrix[..., 1, 2] = 2 * (yz - wx)

    rot_matrix[..., 2, 0] = 2 * (xz - wy)
    rot_matrix[..., 2, 1] = 2 * (yz + wx)
    rot_matrix[..., 2, 2] = 1 - 2 * (xx + yy)

    return rot_matrix
