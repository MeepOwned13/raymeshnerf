import pycolmap as pc
from pathlib import Path
import numpy as np
import torch
from tempfile import TemporaryDirectory
from torchvision.transforms.functional import to_pil_image
from contextlib import contextmanager


pc_img_to_torch_num = lambda img: int(img.name.split(".")[0])


@contextmanager
def with_pycolmap_min_logging_level(level: int = 2):
    original = pc.logging.minloglevel
    try:
        pc.logging.minloglevel = level
        yield
    finally:
        pc.logging.minloglevel = original


def get_sparse_sfm_depths(images: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get sparse depths and errors per pixel from a temporarily built sparse COLMAP reconstruction

    Args:
        images (shape[N, H, W, 3]): Images
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        intrinsics (shape[N, 3, 3]): Intrinsic camera matrices

    Returns:
        tuple: tuple containing (depths, errors)
        - **depths**: *shape[N, H, W]*: Depth at corresponding pixel, set to -1 for pixels with no reconstruction
        - **errors**: *shape[N, H, W]*: Error at corresponding pixel, set to torch.inf for pixels with no reconstruction
    """
    pc.logging.minloglevel = 2

    with TemporaryDirectory() as temp_dir, with_pycolmap_min_logging_level(2):
        temp_dir = Path(temp_dir).resolve()
        rec = get_sparse_colmap_reconstruction(temp_dir, images, extrinsics, intrinsics)
    
    depths, errors = get_sparse_depths_from_colmap_reconstruction(rec, images, extrinsics)
    return depths, errors


def get_sparse_colmap_reconstruction(directory: Path, images: torch.Tensor, extrinsics: torch.Tensor,
                                     intrinsics: torch.Tensor) -> pc.Reconstruction:
    """Get sparse reconstruction using COLMAP
    
    Args:
        directory: Working directory, useful if one wants to save the reconstruction
        images (shape[N, H, W, 3]): Images
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        intrinsics (shape[N, 3, 3]): Intrinsic camera matrices

    Returns:
        rec: Reconstruction populated with sparse reconstruction results
    """
    img_dir = directory / "images"
    img_dir.mkdir()

    # Save images to temp
    for i, image in enumerate(images):
        # Not saving transparency as colmap doesn't handle it anyway, if alpha channel exists it is filtered later
        to_pil_image(image[..., :3].permute(2, 0, 1)).save(img_dir / f"{i:04d}.bmp")
    
    # Feature extraction
    db_path = directory / "colmap.db"
    pc.extract_features(
        database_path=db_path, image_path=img_dir, camera_model="PINHOLE", camera_mode=pc.CameraMode.SINGLE
    )

    # Set camera params
    fx, fy, cx, cy = intrinsics[0, 0, 0], intrinsics[0, 1, 1], intrinsics[0, 0, 2], intrinsics[0, 1, 2]
    cam = pc.Camera(
        camera_id=1,
        model=pc.CameraModelId.PINHOLE,
        width=1600,
        height=1200,
        params=[fx.item(), fy.item(), cx.item(), cy.item()],
    )
    with pc.Database(db_path) as db:
        db.update_camera(cam)

        # Add pose priors from extrinsics
        for img in db.read_all_images():
            extr = extrinsics[pc_img_to_torch_num(img), :3, -1].numpy()
            db.write_pose_prior(
                img.image_id, pc.PosePrior(extr, pc.PosePriorCoordinateSystem.CARTESIAN)
            )

    # Feature matching
    pc.match_exhaustive(db_path)

    # Populate reconstruction params from db (cams, rigs, frames, images)
    rec = pc.Reconstruction()
    with pc.Database(db_path) as db:
        for stuff in db.read_all_cameras():
            rec.add_camera(stuff)

        for stuff in db.read_all_rigs():
            rec.add_rig(stuff)

        for stuff in db.read_all_frames():
            stuff.rig = rec.rig(1)
            img = db.read_image(list(stuff.data_ids)[0].id)
            img = pc_img_to_torch_num(img)
            # Colmap uses y down, z forward, also needs w2c instead of c2w
            c2w = extrinsics[img].clone().numpy()
            c2w[:, 1] *= -1
            c2w[:, 2] *= -1
            w2c = np.linalg.inv(c2w)[:3, :4]
            stuff.set_cam_from_world(1, pc.Rigid3d(w2c))
            rec.add_frame(stuff)

        for stuff in db.read_all_images():
            stuff.frame_id = stuff.image_id
            rec.add_image(stuff)
            rec.register_image(stuff.frame_id)

    # Find sparse reconstruction points
    pc.triangulate_points(
        rec, db_path, img_dir, directory / "output"
    )

    return rec


def get_sparse_depths_from_colmap_reconstruction(rec: pc.Reconstruction, images: torch.Tensor, extrinsics: torch.Tensor
                                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get sparse depths and errors per pixel from a sparse COLMAP reconstruction

    Args:
        rec: Reconstruction populated with sparse reconstruction results
        images (shape[N, H, W, 3]): Images
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)

    Returns:
        tuple: tuple containing (depths, errors)
        - **depths**: *shape[N, H, W]*: Depth at corresponding pixel, set to -1 for pixels with no reconstruction
        - **errors**: *shape[N, H, W]*: Error at corresponding pixel, set to torch.inf for pixels with no reconstruction
    """
    has_alpha = images.shape[-1] > 3
    depths = torch.full(images.shape[:-1], -1, dtype=torch.float32)
    errors = torch.full(images.shape[:-1], torch.inf, dtype=torch.float32)

    for img in rec.images.values():
        torch_img_num = pc_img_to_torch_num(img)
        origin = extrinsics[torch_img_num, :3, -1]

        for p2 in img.points2D:
            if p2.point3D_id - 2**64 == -1:  # -1 id (no point3D) is represented in uint as 2**64 - 1
                continue

            p3 = rec.points3D[p2.point3D_id]
            if np.any(np.abs(p3.xyz) > 1.0) or (p3.error > 2.0):  # out of observed bbox or large reprojection error
                continue

            x, y = torch.tensor(p2.xy).round().to(torch.int32) - 1
            dist = torch.sqrt(torch.sum(torch.pow(origin - torch.tensor(p3.xyz, dtype=torch.float32), 2), -1))

            # Update to new point with less error and ignore anything with alpha=0
            if errors[torch_img_num, y, x] > p3.error and (not has_alpha or images[torch_img_num, y, x, -1] > 0):
                errors[torch_img_num, y, x] = p3.error
                depths[torch_img_num, y, x] = dist

    return depths.unsqueeze(-1), errors.unsqueeze(-1)
