from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import re

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DATA_FOLDER = REPO_ROOT / "data" / "embedding" / "raw_surface_data"


def is_freesurfer_path(path):
    pattern = r"^(.*\/)?(lh|rh)\.(pial|inflated|smoothwm|white|sphere|sphere\.reg|sulc|curv|thickness|area)$"
    return re.search(pattern, path) is not None


@dataclass
class Surface:
    mesh: str = "fs_lr32"
    surface_name: str = "inflated"
    hemisphere: str = "l"
    specie: str = "human"
    custom_file_path: Path = None
    freesurfer_file: bool = False

    def __post_init__(self):
        if self.custom_file_path is not None:
            file_path = str(self.custom_file_path).split("/")[-1]
            if is_freesurfer_path(file_path):
                self.freesurfer_file = True
                self.hemisphere = file_path.split(".")[0][0]
                self.mesh = "individual_mesh"
                self.surface_name = file_path.split(".")[1]
            else:
                self.freesurfer_file = False
                self.hemisphere = file_path.split(".")[1][0]
                self.mesh = file_path.split(".")[0]
                self.surface_name = file_path.split(".")[2]

    @property
    def mesh_folder(self):
        return BRAIN_DATA_FOLDER / self.specie / self.mesh

    @property
    def surface_folder(self):
        return self.mesh_folder / "surf"

    @property
    def file_name(self):
        return f"{self.mesh}.{self.hemisphere}.{self.surface_name}.surf.gii"

    @property
    def file_path(self):
        if self.custom_file_path is not None:
            return self.custom_file_path
        return self.surface_folder / self.file_name

    @cached_property
    def raw_data(self):
        if self.freesurfer_file:
            return nib.freesurfer.io.read_geometry(str(self.file_path))
        gifti_image = nib.load(str(self.file_path))
        return gifti_image.agg_data()

    @property
    def positions(self):
        return self.raw_data[0]

    @property
    def vertex_number(self):
        return self.positions.shape[0]

    @property
    def triangles(self):
        return self.raw_data[1]

    def get_filtered_positions(self, vertex_mask):
        return self.positions[vertex_mask]

    def get_filtered_triangles(self, vertex_mask):
        old_to_new_indices = np.zeros(self.vertex_number, dtype=int) - 1
        old_to_new_indices[vertex_mask] = np.arange(np.sum(vertex_mask))

        filtered_triangles = []
        for tri in self.triangles:
            if all(old_to_new_indices[tri] >= 0):
                filtered_triangles.append(old_to_new_indices[tri])

        return np.array(filtered_triangles, dtype=np.int32)