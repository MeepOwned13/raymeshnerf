import mitsuba as mi
import numpy as np
from pathlib import Path
import argparse

for variant in (['cuda_ad_rgb', 'llvm_ad_rgb', 'scalar_rgb']):
    try:
        mi.set_variant(variant)
        break
    except ImportError:
        pass
from mitsuba import ScalarTransform4f as ST


def load_and_normalize_mesh(obj_path: Path) -> mi.Mesh:
    """Loads and normalizes mesh to [-1,1] bbox

    Args:
        obj_path: Path to the object root

    Returns:
        mesh: Normalized mesh
    """
    mesh: mi.Mesh = mi.load_dict({
        'type': 'obj',
        'filename': (obj_path / 'meshes/model.obj').as_posix(),
    })

    bbox = mesh.bbox()  # Used for re-centering and scaling to -1:1 bounding box

    mesh: mi.Mesh = mi.load_dict({
        'type': 'obj',
        'filename': (obj_path / 'meshes/model.obj').as_posix(),
        'to_world': ST().scale(1 / max(abs(bbox.max - bbox.min) / 2)).translate(-(bbox.max + bbox.min) / 2),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': (obj_path / 'materials/textures/texture.png').as_posix(),
                'wrap_mode': 'clamp',
                'filter_type': 'bilinear',
            }
        }
    })

    return mesh


def sensor_c2w(sensor: mi.Sensor) -> np.ndarray:
    """Converts mitsuba sensor to Extrinsic camera matrix

    Args:
        sensor: Mitsuba sensor to convert

    Returns:
        transformation(shape[4, 4]): Extrinsic camera matrix
    """
    transf = np.array(sensor.world_transform().matrix, dtype=np.float32)
    if len(transf.shape) > 2:  # On CUDA its dim=3, on Scalar its dim=2
        transf = transf[:, :, 0]
    # -1 mult to z axis as mitsuba seems to represent camera outbound direction as [0,0,1]
    transf[:3, 2] *= -1
    return transf


def create_batch_sensor(n: int, radius: float, size: int = 800, fov_x: float = 40, deterministic=False) -> mi.Sensor:
    """Creates a Mitsuba batch sensor

    Args:
        n: Number of view directions (ideally power of 2)
        radius: Distance from object center
        size: Size of rectangle sides for the cameras
        fov_x: X axis fov in degrees for cameras
        deterministic: If False, modulates view directions a little

    Returns:
        batch_sensor: Mitsuba batch sensor, rendered image is of shape(size, n * size)
    """
    focal = (size / 2) / np.tan(np.deg2rad(fov_x) / 2)

    i = np.arange(0, n, dtype=float) + 0.5
    if not deterministic:
        i = i + np.random.uniform(-0.5, 0.5, size = n) / n
        
    phis = np.pi * i * (1 + np.sqrt(5))
    thetas = np.arccos(1 - 2 * i / n)

    sensors: list[mi.Sensor] = [mi.load_dict({
        'type': 'perspective',
        'fov': fov_x,
        'fov_axis': 'x',
        'to_world': ST().look_at(
            origin=[
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            target=[0, 0, 0],
            up=[0, 0, 1],  # Mitsuba 3 uses z as the vertical axis
        )
    }) for phi, theta in zip(phis, thetas)]

    extrinsics: np.ndarray = np.stack([sensor_c2w(s) for s in sensors], axis=0)

    batch_sensor = {
        'type': 'batch',
        'sampler': {
            'type': 'ldsampler',
            'sample_count': 64,
        },
        'film': {
            'type': 'hdrfilm',
            'width': size * len(sensors),
            'height': size,
            'pixel_format': 'rgba',
            'filter': {
                'type': 'tent'
            }
        },
    }
    batch_sensor.update({f's{i}': s for i, s in enumerate(sensors)})

    return mi.load_dict(batch_sensor), extrinsics, focal.astype(np.float32)


def render_mesh(obj_path: Path, sensor_count: int, radius: float = 4.0, size: int = 200, fov_x: float = 40,
                deterministic=False) -> tuple[np.ndarray, np.ndarray, float]:
    """Renders mesh specified by path from multiple angles

    Args:
        obj_path: Path to the object root
        sensor_count: Number of view directions (ideally power of 2)
        radius: Distance from object center
        size: Size of rectangle sides for the cameras
        fov_x: X axis fov in degrees for cameras
        deterministic: If False, modulates view directions a little

    Returns:
        tuple: tuple containing (images, extrinsics, focal)
        - **images**: *shape[sensor_count, size, size, 3]*: Rendered images
        - **extrinsics**: *shape[sensor_count, 4, 4]*: Extrinsic camera matrices
        - **focal**: Focal length of cameras
    """
    mesh = load_and_normalize_mesh(obj_path)

    scene_dict: dict = {
        'type': 'scene',
        'integrator': {
            'type': 'direct'
        },
        'obj': mesh,
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 1.0,
            }
        },
    }

    scene: mi.Scene = mi.load_dict(scene_dict)

    sensor, extrinsics, focal = create_batch_sensor(
        n=sensor_count,
        radius=radius,
        size=size,
        fov_x=fov_x,
        deterministic=deterministic,
    )

    render = np.asarray(mi.render(scene, sensor=sensor), dtype=np.float32).clip(0, 1)
    # This is complicated because srgb conversion is done here too (saving to disk would do this too)
    # If saving to disk is the goal, this shouldn't be done as the conversion would happen again
    # This is bypassed by converting here, then saving as .npz which doesn't do a conversion
    images = np.asarray(mi.Bitmap(render).convert(srgb_gamma=True, component_format=mi.Struct.Type.Float32))
    images = images.reshape(size, sensor_count, size, -1).transpose(1, 0, 2, 3)

    return images, extrinsics, focal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-on', '--object_name', type=str, required=True)
    parser.add_argument('-sc', '--sensor_count', type=int, default=64)
    parser.add_argument('-s', '--size', type=int, default=200)
    args = parser.parse_args()

    obj_path = (Path(__file__) / "../../../data/raw_objects" / args.object_name).resolve().absolute()
    save_path = (Path(__file__) / "../../../data" / args.object_name).resolve().absolute()

    images, c2ws, focal = render_mesh(obj_path, args.sensor_count)
    np.savez_compressed(save_path, images=images, c2ws=c2ws, focal=focal)

