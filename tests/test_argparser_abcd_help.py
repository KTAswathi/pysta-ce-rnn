"""Focused checks for ABCD command-line applicability and task defaults."""

import sys

import pytest

from pysta.argparser import parse_args


def test_abcd_changed_defaults_are_task_conditioned_and_overrideable(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])

    maze = parse_args(task="maze")
    abcd = parse_args(task="abcd_fmri")
    overridden = parse_args(task="abcd_fmri", local_fraction=0.3)

    assert maze["local_fraction"] == pytest.approx(1.0 / 6.0)
    assert abcd["local_fraction"] == pytest.approx(1.0 / 4.0)
    assert overridden["local_fraction"] == pytest.approx(0.3)
    assert maze["lrate"] == pytest.approx(3e-4)
    assert abcd["lrate"] == pytest.approx(1e-4)


def test_help_exposes_abcd_applicability_and_ignored_legacy_options(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["train", "--help"])
    with pytest.raises(SystemExit, match="0"):
        parse_args()

    help_text = " ".join(capsys.readouterr().out.split())
    for tag in (
        "[ABCD + maze]",
        "[ABCD task]",
        "[ABCD cortical/line-embedded]",
        "[ABCD cortical]",
        "[ABCD: routing ignored]",
        "[ABCD: training ignored]",
        "[maze only; ignored by ABCD]",
    ):
        assert tag in help_text

    assert "legacy maze-routing flag" in help_text
    assert "training constructor does not consume this option" in help_text
