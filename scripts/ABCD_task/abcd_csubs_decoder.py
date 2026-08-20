"""Joint multinomial Csubs decoder used by the ABCD analysis.

This module isolates the original repository's Csubs optimization from the
superseded stage-based analysis script.  The primary ``raw_dot`` mode preserves
the historical objective exactly; ``normalized`` is the pre-specified
cosine-overlap sensitivity. Keeping the fitter here makes the normalized ABCD
analysis self-contained and the superseded stage-based script removable.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F


NUM_LOCATIONS = 9


def make_overlap_diagonal_mask(
    *, n_lags: int, n_locations: int, device: torch.device
) -> torch.Tensor:
    """Mask only same-horizon, same-class self-overlap terms."""

    mask = torch.ones(
        (n_lags, n_lags, n_locations, n_locations),
        dtype=torch.float32,
        device=device,
    )
    lag_idx = torch.arange(n_lags, device=device)
    location_idx = torch.arange(n_locations, device=device)
    mask[
        lag_idx[:, None],
        lag_idx[:, None],
        location_idx[None, :],
        location_idx[None, :],
    ] = 0.0
    return mask


def calculate_overlap(
    coeffs: torch.Tensor,
    *,
    mode: str,
    diagonal_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute class/horizon filter overlap for the selected estimator."""

    if mode == "raw_dot":
        overlap_coeffs = coeffs
    elif mode == "normalized":
        denominator = torch.sqrt(
            torch.sum(coeffs**2, dim=-1, keepdim=True) + 1e-12
        )
        overlap_coeffs = coeffs / denominator
    else:
        raise ValueError("overlap mode must be 'raw_dot' or 'normalized'.")

    overlap = torch.einsum(
        "lcu,mdu->lmcd", overlap_coeffs, overlap_coeffs
    )
    return overlap * diagonal_mask


def fit_original_joint_csubs(
    *,
    rs: np.ndarray,
    future_locations: np.ndarray,
    future_valid: np.ndarray,
    future_lags: np.ndarray,
    L2_alpha: float,
    overlap_alpha: float,
    L1_alpha: float,
    warmup: int,
    max_iters: int,
    atol: float,
    rtol: float,
    lrate: float,
    fit_seed: int,
    overlap_mode: str,
    device_name: str,
) -> dict[str, np.ndarray | float | int | str]:
    """Fit all future-location decoders jointly.

    The optimizer, loss, warm-up, stopping rule, and output normalization are
    intentionally the historical implementation.  ABCD-specific cross-fitting,
    class gauge handling, inactive-unit handling, and map construction remain in
    :mod:`analyse_abcd_normalized_csubs`.
    """

    n_samples, n_units = rs.shape
    n_lags = len(future_lags)
    n_locations = NUM_LOCATIONS

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("\n======================================")
    print("ORIGINAL JOINT CSUBS FIT")
    print("======================================")
    print(f"device: {device}")
    print(f"subspaces/lags: {n_lags}")
    print(f"locations/classes: {n_locations}")
    print(f"neurons: {n_units}")
    print(f"navigation observations: {n_samples}")
    print(f"overlap mode: {overlap_mode}")
    print("\nHyperparameters:")
    print(f"  L2_alpha      = {L2_alpha}")
    print(f"  L1_alpha      = {L1_alpha}")
    print(f"  overlap_alpha = {overlap_alpha}")
    print(f"  warmup         = {warmup}")
    print(f"  max_iters      = {max_iters}")
    print(f"  atol            = {atol}")
    print(f"  rtol            = {rtol}")
    print(f"  lrate           = {lrate}")
    print(f"  fit_seed        = {fit_seed}")

    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)

    X = torch.tensor(rs, dtype=torch.float32, device=device)
    labels_np = np.asarray(future_locations, dtype=int).T
    valid_np = np.asarray(future_valid, dtype=bool).T
    labels_safe_np = labels_np.copy()
    labels_safe_np[~valid_np] = 0
    labels = torch.tensor(labels_safe_np, dtype=torch.long, device=device)
    valid = torch.tensor(valid_np, dtype=torch.float32, device=device)
    valid_counts = torch.sum(valid, dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("At least one horizon has no valid samples.")

    coeffs = torch.nn.Parameter(
        torch.randn(
            (n_lags, n_locations, n_units),
            dtype=torch.float32,
            device=device,
        )
        / np.sqrt(n_units),
        requires_grad=True,
    )
    biases = torch.nn.Parameter(
        torch.zeros(
            (n_lags, n_locations, 1), dtype=torch.float32, device=device
        ),
        requires_grad=True,
    )
    diagonal_mask = make_overlap_diagonal_mask(
        n_lags=n_lags, n_locations=n_locations, device=device
    )

    def calculate_loss(current_overlap_alpha: float):
        logits = torch.einsum("lcu,nu->lcn", coeffs, X) + biases
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = torch.gather(
            log_probs, dim=1, index=labels[:, None, :]
        ).squeeze(dim=1)
        nll_per_lag = -torch.sum(selected_log_probs * valid, dim=1) / valid_counts
        decoding_loss = torch.sum(nll_per_lag)
        regularization_loss = L2_alpha * torch.sum(coeffs**2)
        regularization_loss = regularization_loss + L1_alpha * torch.sum(
            torch.abs(coeffs)
        )
        overlap = calculate_overlap(
            coeffs, mode=overlap_mode, diagonal_mask=diagonal_mask
        )
        overlap_loss = current_overlap_alpha * torch.sum(torch.abs(overlap))
        total = decoding_loss + regularization_loss + overlap_loss
        return total, decoding_loss, regularization_loss, overlap_loss, logits

    optimizer = torch.optim.Adam(
        [coeffs, biases], lr=lrate, betas=(0.99, 0.999)
    )
    loss_history = np.full(max_iters, np.nan, dtype=float)
    decoding_history = np.full(max_iters, np.nan, dtype=float)
    reg_history = np.full(max_iters, np.nan, dtype=float)
    overlap_history = np.full(max_iters, np.nan, dtype=float)
    warmup_scale_history = np.full(max_iters, np.nan, dtype=float)
    minimum_iters = max(40, int(warmup * 1.5))
    old_avg = np.inf
    new_avg = np.inf
    stopped_at = max_iters

    for iteration in range(max_iters):
        warmup_scale = max(
            0.0,
            min(
                1.0,
                (iteration - 0.2 * warmup) / (0.8 * warmup),
            ),
        )
        current_overlap_alpha = warmup_scale * overlap_alpha
        optimizer.zero_grad()
        (
            total_loss,
            decoding_loss,
            regularization_loss,
            overlap_loss,
            logits,
        ) = calculate_loss(current_overlap_alpha)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}.")
        total_loss.backward()
        optimizer.step()

        loss_history[iteration] = float(total_loss.detach().cpu())
        decoding_history[iteration] = float(decoding_loss.detach().cpu())
        reg_history[iteration] = float(regularization_loss.detach().cpu())
        overlap_history[iteration] = float(overlap_loss.detach().cpu())
        warmup_scale_history[iteration] = warmup_scale

        if iteration >= 40:
            old_avg = np.mean(loss_history[iteration - 40 : iteration - 20])
            new_avg = np.mean(loss_history[iteration - 20 : iteration])

        if iteration % 50 == 0 or iteration == max_iters - 1:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                training_accuracies = []
                for lag_i in range(n_lags):
                    use = valid[lag_i] > 0
                    acc = torch.mean(
                        (predictions[lag_i, use] == labels[lag_i, use]).float()
                    )
                    training_accuracies.append(float(acc.cpu()))
            print(
                f"iter={iteration:4d} | loss={loss_history[iteration]:.5f} | "
                f"decode={decoding_history[iteration]:.5f} | "
                f"reg={reg_history[iteration]:.5f} | "
                f"overlap={overlap_history[iteration]:.5f} | "
                f"warmup={warmup_scale:.3f}"
            )
            print("  train acc: " + " ".join(f"{x:.2f}" for x in training_accuracies))
            sys.stdout.flush()

        if iteration + 1 >= minimum_iters and iteration >= 40:
            improvement = old_avg - new_avg
            denominator = max(abs(loss_history[iteration]), 1e-12)
            relative_improvement = improvement / denominator
            should_continue = improvement > atol and relative_improvement > rtol
            if not should_continue:
                stopped_at = iteration + 1
                print("\nConvergence criterion reached:")
                print(f"  iteration: {stopped_at}")
                print(f"  old rolling loss: {old_avg:.6f}")
                print(f"  new rolling loss: {new_avg:.6f}")
                print(f"  improvement: {improvement:.6g}")
                print(f"  relative improvement: {relative_improvement:.6g}")
                break

    with torch.no_grad():
        (
            final_total,
            final_decoding,
            final_reg,
            final_overlap,
            final_logits,
        ) = calculate_loss(overlap_alpha)
        final_predictions = torch.argmax(final_logits, dim=1)
        training_accuracy = np.zeros(n_lags, dtype=float)
        for lag_i in range(n_lags):
            use = valid[lag_i] > 0
            training_accuracy[lag_i] = float(
                torch.mean(
                    (final_predictions[lag_i, use] == labels[lag_i, use]).float()
                ).cpu()
            )

    Csubs_raw = coeffs.detach().cpu().numpy()
    biases_np = biases.detach().cpu().numpy()
    coefficient_norms = np.sqrt(
        np.sum(Csubs_raw**2, axis=-1, keepdims=True)
    )
    if np.any(coefficient_norms < 1e-12):
        raise ValueError("At least one location decoder has near-zero coefficient norm.")
    Csubs = Csubs_raw / coefficient_norms
    Csubs_score = np.linalg.norm(Csubs, axis=1)
    Csubs_raw_score = np.linalg.norm(Csubs_raw, axis=1)

    if Csubs.shape != (n_lags, n_locations, n_units):
        raise RuntimeError(f"Unexpected Csubs shape: {Csubs.shape}")
    if Csubs_score.shape != (n_lags, n_units):
        raise RuntimeError(f"Unexpected Csubs_score shape: {Csubs_score.shape}")
    normalized_decoder_norms = np.sqrt(np.sum(Csubs**2, axis=-1))
    if not np.allclose(normalized_decoder_norms, 1.0, atol=1e-5):
        raise RuntimeError("Normalized Csubs location vectors do not have unit norm.")
    score_energy = np.sum(Csubs_score**2, axis=1)
    if not np.allclose(score_energy, float(n_locations), atol=1e-4):
        raise RuntimeError(
            "Csubs score-energy invariant failed.\n"
            f"Expected {n_locations}, got {score_energy}"
        )

    used_history = slice(0, stopped_at)
    return {
        "Csubs": Csubs,
        "Csubs_raw": Csubs_raw,
        "biases": biases_np,
        "Csubs_score": Csubs_score,
        "Csubs_raw_score": Csubs_raw_score,
        "coefficient_norms_raw": coefficient_norms[..., 0],
        "training_accuracy": training_accuracy,
        "score_energy": score_energy,
        "loss_history": loss_history[used_history],
        "decoding_loss_history": decoding_history[used_history],
        "regularization_loss_history": reg_history[used_history],
        "overlap_loss_history": overlap_history[used_history],
        "warmup_scale_history": warmup_scale_history[used_history],
        "iterations_run": int(stopped_at),
        "final_total_loss": float(final_total.cpu()),
        "final_decoding_loss": float(final_decoding.cpu()),
        "final_regularization_loss": float(final_reg.cpu()),
        "final_overlap_loss": float(final_overlap.cpu()),
        "device": str(device),
    }
