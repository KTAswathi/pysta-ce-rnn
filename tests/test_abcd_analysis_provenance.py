"""Focused provenance checks for cortical ABCD mechanism sweeps."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.ABCD_task.abcd_analysis_common import (
    build_cortical_mechanism_manifest,
    resolve_anchor_reference_units,
)


class _CorticalModelStub:
    Nrec = 8
    Nin = 5
    local_fraction = 0.5
    dist_reg = 1e-7
    line_decay = 0.12
    use_local_init = True
    line_init_scale = 1.0
    readout_mode = "global"
    input_routing_modes = {
        "current_location": "same_end",
        "instruction_location": "same_end",
        "phase": "global",
    }

    def __init__(self, embedding_dir):
        self._embedding_directory = embedding_dir
        self.env = type(
            "EnvStub",
            (),
            {"obs_inds": lambda _: {
                "current_location": np.asarray([0, 1]),
                "instruction_location": np.asarray([2, 3]),
                "phase": np.asarray([4]),
            }},
        )()
        self.same_end_unit_mask = torch.tensor(
            [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32
        )
        self.anatomical_anchor_unit_indices = [1, 3]
        self.input_mask_buffers = {
            "current_location": "mask_current_location",
            "instruction_location": "mask_instruction_location",
            "phase": "mask_phase",
        }
        local = self.same_end_unit_mask[:, None]
        self.mask_current_location = torch.zeros(self.Nrec, self.Nin)
        self.mask_current_location[:, :2] = local
        self.mask_instruction_location = torch.zeros(self.Nrec, self.Nin)
        self.mask_instruction_location[:, 2:4] = local
        self.mask_phase = torch.zeros(self.Nrec, self.Nin)
        self.mask_phase[:, 4] = 1

    def _embedding_dir(self):
        return self._embedding_directory


def test_manifest_separates_fixed_anchor_from_expanded_input_zone(tmp_path):
    np.save(tmp_path / "anchor_unit_indices.npy", np.asarray([3, 1], dtype=np.int32))
    provenance = build_cortical_mechanism_manifest(_CorticalModelStub(tmp_path))

    assert provenance["resolved_parameters"] == {
        "local_fraction": 0.5,
        "dist_reg": 1e-7,
        "line_decay": 0.12,
        "use_local_init": True,
        "line_init_scale": 1.0,
        "readout_mode": "global",
    }
    recipient = provenance["local_input_recipient_zone"]
    seed = provenance["anatomical_anchor_seed"]
    reference = provenance["anchor_distance_reference"]
    assert recipient["unit_count"] == 4
    assert recipient["unit_indices"].tolist() == [0, 1, 2, 3]
    assert recipient["observation_groups"] == [
        "current_location",
        "instruction_location",
    ]
    assert seed["unit_count"] == 2
    assert seed["unit_indices"].tolist() == [1, 3]
    assert reference["unit_indices"].tolist() == [1, 3]
    assert reference["uses_expanded_input_recipient_zone"] is False

    manifest = {"cortical_mechanism": provenance}
    resolved = resolve_anchor_reference_units(manifest, tmp_path, n_units=8)
    np.testing.assert_array_equal(resolved, [1, 3])


def test_manifest_rejects_tampered_recipient_zone(tmp_path):
    np.save(tmp_path / "anchor_unit_indices.npy", np.asarray([1, 3], dtype=np.int64))
    provenance = build_cortical_mechanism_manifest(_CorticalModelStub(tmp_path))
    broken = deepcopy(provenance)
    broken["local_input_recipient_zone"]["unit_mask"][0] = 0

    with pytest.raises(ValueError, match="mask and indices disagree"):
        resolve_anchor_reference_units(
            {"cortical_mechanism": broken}, tmp_path, n_units=8
        )
