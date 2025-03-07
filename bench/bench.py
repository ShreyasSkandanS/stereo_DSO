from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import subprocess as sp
import shutil


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


class Runner:

    def __init__(self) -> None:
        pass

    def run(self):
        pass

    def save(self):
        pass


class SDSO(Runner):

    def __init__(self, base_dir, dataset_name, data_prefix, files, calib, preset, mode, reverse):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.files = base_dir / dataset_name / files
        self.calib = calib
        self.preset = preset
        self.mode = mode
        self.reverse = reverse
        self.prefix = data_prefix
        self.nomt = 1
        self.quiet = 1
        self.gt = Path().resolve().parent / 'groundTruthPose/dummy.txt'

        self.output_dir = base_dir / dataset_name / 'SDSO'
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.cmd = ['/tmp/dso_dataset', 'files=' + str(self.files), 'calib=' + str(self.calib),
                    'groundtruth=' + str(self.gt), 'mode=' + str(self.mode), 'nogui=1',
                    'quiet=' + str(self.quiet), 'nolog=1', 'nomt=' + str(self.nomt)]
        if self.reverse:
            self.cmd.append('reverse=1')

    def run(self):
        sp.run(self.cmd)

    def save(self):
        save_path = self.output_dir / str(self.prefix).replace('/', '_')
        if self.reverse:
            save_path = f'{save_path}_rev.txt'
        else:
            save_path = f'{save_path}_fwd.txt'
        shutil.copy('result.txt', save_path)


class DSOL(Runner):

    def __init__(self, base_dir, dataset_name, dataset_id, data_prefix, files, reverse):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.files = base_dir / dataset_name / files
        self.reverse = reverse
        self.prefix = data_prefix
        self.log = 0
        self.tbb = 0
        self.viz = False
        self.wait_ms = 0

        self.output_dir = base_dir / dataset_name / 'DSOL'
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.save_path = self.output_dir / str(self.prefix).replace('/', '_')
        if self.reverse:
            self.save_path = f'{self.save_path}_rev.txt'
        else:
            self.save_path = f'{self.save_path}_fwd.txt'

        self.cmd = ['roslaunch', 'svcpp', 'dsol_data.launch', 'save:=' + str(self.save_path),
                    'data_dir:=' + str(self.files), 'data:=' + dataset_id, 'log:=' + str(self.log),
                    'tbb:=' + str(self.tbb), 'vis:=' + str(self.viz), 'wait_ms:=' + str(self.wait_ms)]
        if self.reverse:
            self.cmd.append('reverse:=True')

    def run(self):
        sp.run(self.cmd)


class Dataset:

    def __init__(self, base_dir, dataset_name) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_name = dataset_name
        self.output_dir().mkdir(exist_ok=True, parents=True)
        self.data_dirs = []

        print(self.dataset_name)

    def __len__(self) -> int:
        return len(self.data_dirs)

    def get_name(self, idx: int) -> str:
        raise NotImplementedError("Not Implemented")

    def get_gt_file(self, idx: int) -> Path:
        raise NotImplementedError("Not Implemented")

    def output_dir(self) -> Path:
        return self.base_dir / self.dataset_name / 'gt'

    def get_pose(self, idx: int) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    def write_all_gt(self):
        for i in range(len(self)):
            self.write_single_gt(i)

    def write_single_gt(self,i):
        write_tum_fwd_rev(self.get_pose(i),
                          f"{self.output_dir()}/{self.get_name(i)}_fwd.txt",
                          f"{self.output_dir()}/{self.get_name(i)}_rev.txt")

class KittiDataset(Dataset):
    scenes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    def __init__(self, base_dir: str = "/tmp"):
        super().__init__(base_dir, 'kitti')
        self.data_dirs = [
            f'{scene}' for scene in self.scenes
        ]

        repository_path = Path().resolve().parent
        self.calib = repository_path / 'calib' / self.dataset_name

        print(self.data_dirs)

    def get_name(self, i: int) -> str:
        return self.data_dirs[i].replace('/', '_')

    def get_sdso(self, i: int, reverse: bool = False) -> SDSO:
        return SDSO(self.base_dir,
                    self.dataset_name,
                    self.data_dirs[i],
                    f'sequences/{self.data_dirs[i]}',
                    f'{self.calib}/{self.data_dirs[i]}.txt',
                    preset=0,
                    mode=1,
                    reverse=reverse)

    def get_dsol(self, i: int, reverse: bool = False) -> DSOL:
        return DSOL(self.base_dir,
                    self.dataset_name,
                    'kit',
                    self.data_dirs[i],
                    f'sequences/{self.data_dirs[i]}',
                    reverse=reverse)

    def get_gt_file(self, i: int) -> Path:
        return self.base_dir / self.dataset_name / "poses" / f'{self.data_dirs[i]}.txt'

    def get_pose(self, i: int) -> np.ndarray:
        file = self.get_gt_file(i)
        print(f"Reading from {file}")
        poses = self.prep_poses(file)
        return poses

    @staticmethod
    def prep_poses(file: Path) -> np.ndarray:
        data = np.loadtxt(file)
        last_row = np.tile([0, 0, 0, 1], (data.shape[0], 1))
        data = np.hstack([data, last_row])
        data = data.reshape(-1, 4, 4)
        data[:, :3, :3] = Rotation.from_matrix(
            data[:, :3, :3]).as_matrix()
        return data


class TartanAirDataset(Dataset):
    scenes = ["carwelding", "gascola", "office", "oldtown"]

    # scenes = ["carwelding", "gascola", "office", "oldtown", "office2", "hospital"]

    def __init__(self, base_dir: str = "/tmp"):
        super().__init__(base_dir, 'tartan_air')

        self.data_dirs = []
        for scene in self.scenes:
            for p in Path(f'{self.base_dir}/{self.dataset_name}/{scene}/Easy/').iterdir():
                if p.is_dir() and p.stem[0] == 'P':
                    self.data_dirs.append(f'{scene}/Easy/{p.stem}')

        repository_path = Path().resolve().parent
        self.calib = repository_path / 'calib' / self.dataset_name / 'tartan_air.txt'

        print(self.data_dirs)

    def get_name(self, i: int) -> str:
        return self.data_dirs[i].replace('/', '_')

    def get_sdso(self, i: int, reverse: bool = False) -> SDSO:
        return SDSO(self.base_dir,
                    self.dataset_name,
                    self.data_dirs[i],
                    self.data_dirs[i],
                    self.calib,
                    preset=0,
                    mode=2,
                    reverse=reverse)

    def get_dsol(self, i: int, reverse: bool = False) -> DSOL:
        return DSOL(self.base_dir,
                    self.dataset_name,
                    'tta',
                    self.data_dirs[i],
                    self.data_dirs[i],
                    reverse=reverse)

    def get_gt_file(self, i: int) -> Path:
        return self.base_dir / self.dataset_name / self.data_dirs[i] / 'pose_left.txt'

    def get_pose(self, i: int) -> np.ndarray:
        file = self.get_gt_file(i)
        print(f"Reading from {file}")
        poses = self.prep_poses(file)
        return poses

    @staticmethod
    def prep_poses(pose_file) -> np.ndarray:
        # tx ty tz qx qy qz qw
        data = np.loadtxt(pose_file)
        poses = np.zeros((data.shape[0], 4, 4))
        poses[:, :3, 3] = data[:, :3]
        poses[:, 3, 3] = 1
        # rotation from cam to ned
        R_ned_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

        from scipy.spatial.transform import Rotation
        # motion is defined in ned frame, but what we really need is motion of camera
        # T_w_c = T_w_ned @ R_ned_c
        poses[:, :3, :3] = Rotation.from_quat(data[:, 3:]).as_matrix() @ R_ned_cam
        # normalize pose to the first pose
        # T_c0_ci = T_w_c0^-1 @ T_w_ci = T_c0_w @ T_w_ci
        # poses = np.linalg.inv(poses[0]) @ poses
        return poses


class VkittiDataset(Dataset):
    scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    variations = ["clone", "rain", "15-deg-left", "15-deg-right", "30-deg-left",
                  "30-deg-right", "fog", "morning", "overcast", "sunset"]

    def __init__(self, base_dir: str = "/tmp"):
        super().__init__(base_dir, 'vkitti')
        self.data_dirs = [
            f'{scene}/{variation}' for scene in self.scenes
            for variation in self.variations
        ]

        repository_path = Path().resolve().parent
        self.calib = repository_path / 'calib/vkitti/Scene01.txt'

        print(self.data_dirs)

    def get_name(self, i: int) -> str:
        return self.data_dirs[i].replace('/', '_')

    def get_sdso(self, i: int, reverse: bool = False) -> SDSO:
        return SDSO(self.base_dir,
                    self.dataset_name,
                    self.data_dirs[i],
                    f'{self.data_dirs[i]}/frames/rgb',
                    self.calib,
                    preset=0,
                    mode=2,
                    reverse=reverse)

    def get_dsol(self, i: int, reverse: bool = False) -> DSOL:
        return DSOL(self.base_dir,
                    self.dataset_name,
                    'vk2',
                    self.data_dirs[i],
                    self.data_dirs[i],
                    reverse=reverse)

    def get_gt_file(self, i: int) -> Path:
        return self.base_dir / self.dataset_name / self.data_dirs[i] / "extrinsic.txt"

    def get_pose(self, i: int) -> np.ndarray:
        file = self.get_gt_file(i)
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
