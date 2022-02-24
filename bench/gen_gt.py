from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import typing as T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Inverse transform [R, t]^-1 = [R', -R'@t]"""
    assert T.shape[-2:] == (4, 4), T.shape

    T_inv = np.zeros_like(T)
    R_t = np.swapaxes(T[..., :3, :3], -1, -2)
    T_inv[..., :3, :3] = R_t
    T_inv[..., :3, [3]] = -(R_t @ T[..., :3, [3]])
    T_inv[..., -1, -1] = 1.0

    return T_inv


def poses2tum(poses: np.ndarray) -> np.ndarray:
    # normalize poses
    poses = inverse_transform(poses[[0]]) @ poses
    quats = Rotation.from_matrix(poses[..., :3, :3]).as_quat()
    n = len(poses)

    tum_data = np.empty((n, 8))
    tum_data[:, 0] = np.arange(n)
    tum_data[:, 1:4] = poses[:, :3, 3]
    tum_data[:, 4:] = quats
    return tum_data


def write_tum(tum_data: np.ndarray, filename: str):
    print(f"writing {filename}")
    np.savetxt(filename,
               tum_data,
               fmt=f"%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f")


def write_tum_fwd_rev(poses, filename, filename_rev):
    write_tum(poses2tum(poses), filename)
    write_tum(poses2tum(poses[::-1]), filename_rev)


class Dataset:

    def __init__(self, base_dir) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.files = []
        self.names = []

    def __len__(self) -> int:
        return len(self.files)

    def get_name(self, idx: int) -> str:
        return self.names[idx]

    def get_files(self, idx: int) -> Path:
        return self.files[idx]

    def output_dir(self) -> Path:
        return self.base_dir

    def get_pose(self, idx: int) -> np.ndarray:
        raise NotImplementedError("Not implemented")


class KittiDataset(Dataset):
    pass


class TartanAirDataset(Dataset):
    pass


class VkittiDataset(Dataset):

    scenes = ["Scene01", "Scene02"]
    variations = ["clone", "rain"]

    def __init__(self, base_dir: str = "/tmp/vkitti"):
        super().__init__(base_dir)
        self.files = [
            self.base_dir / scene / variation / "extrinsic.txt"
            for scene in self.scenes for variation in self.variations
        ]

        self.names = [
            f"{scene}_{variation}" for scene in self.scenes
            for variation in self.variations
        ]

        print(self.files)

    def get_pose(self, idx: int) -> np.ndarray:
        file = self.get_file(idx)
        print(f"Reading from {file}")
        extrins = self.prep_extrinsics(file)
        poses = inverse_transform(extrins)
        return poses

    @staticmethod
    def prep_extrinsics(file: Path) -> np.ndarray:
        # i, cam, T00, T01, T02, T03, T11, ...
        data = np.loadtxt(file, skiprows=1)
        extrins = data[:, 2:].reshape(-1, 4, 4)
        # skip every other row
        extrins = extrins[::2]
        # fix rotation using scipy
        extrins[:, :3, :3] = Rotation.from_matrix(
            extrins[:, :3, :3]).as_matrix()
        return extrins


def write_all(ds: Dataset):
    for i in range(len(ds)):
        write_tum_fwd_rev(ds.get_pose(i),
                          f"{ds.output_dir()}/{ds.get_name(i)}_fwd.txt",
                          f"/{ds.output_dir()}/{ds.get_name(i)}_rev.txt")


if __name__ == "__main__":
    vk2 = VkittiDataset()
    # kit = KittiDataset()

    write_all(vk2)
    # write_all(kit)
