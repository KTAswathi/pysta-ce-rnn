import numpy as np

from scripts.ABCD_task.abcd_csubs_decoder import fit_original_joint_csubs
from scripts.ABCD_task.analyse_abcd_normalized_csubs import (
    fit_original_joint_csubs as normalized_analysis_fitter,
)


def test_normalized_analysis_uses_standalone_decoder_module():
    assert normalized_analysis_fitter is fit_original_joint_csubs
    assert fit_original_joint_csubs.__module__.endswith("abcd_csubs_decoder")


def test_original_joint_decoder_shapes_and_score_energy():
    rng = np.random.default_rng(11)
    activity = rng.normal(size=(18, 4)).astype(np.float32)
    labels = np.column_stack(
        (np.arange(18, dtype=np.int64) % 9, (np.arange(18) + 1) % 9)
    )
    fit = fit_original_joint_csubs(
        rs=activity,
        future_locations=labels,
        future_valid=np.ones_like(labels, dtype=bool),
        future_lags=np.arange(2),
        L2_alpha=1e-3,
        overlap_alpha=2e-3,
        L1_alpha=1e-4,
        warmup=2,
        max_iters=2,
        atol=1e-3,
        rtol=2e-4,
        lrate=5e-3,
        fit_seed=19,
        overlap_mode="raw_dot",
        device_name="cpu",
    )

    assert fit["Csubs_raw"].shape == (2, 9, 4)
    assert fit["Csubs_score"].shape == (2, 4)
    np.testing.assert_allclose(fit["score_energy"], np.full(2, 9.0), atol=1e-4)
    assert np.isfinite(fit["final_total_loss"])
