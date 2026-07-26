from __future__ import annotations

import json
from pathlib import Path

import pytest

from minireplay.config import load_config
from minireplay.constants import CONFIG_SCHEMA
from minireplay.errors import ValidationError

FAKE_REPO = Path(__file__).parent / "fake_repo"


def write_config(tmp_path: Path, **updates) -> Path:
    value = {
        "schema_version": CONFIG_SCHEMA,
        "framework": "mini-swe",
        "repo": str(FAKE_REPO),
        "concurrency": 8,
        "duration_s": 60,
        "seed": 42,
        "targets": {"vllm-8000": "http://127.0.0.1:8000"},
        **updates,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("refill", (True, False))
def test_refill_round_trips_and_is_workload_identity(tmp_path: Path, refill: bool) -> None:
    config = load_config(write_config(tmp_path, refill=refill))
    assert config.refill is refill
    assert config.to_json()["refill"] is refill
    assert config.workload_identity()["refill"] is refill
    assert "load_model" not in config.to_json()


@pytest.mark.parametrize(
    ("load_model", "expected"),
    (("steady", True), ("fire-once", False)),
)
def test_legacy_load_model_is_an_input_only_alias(
    tmp_path: Path,
    load_model: str,
    expected: bool,
) -> None:
    config = load_config(write_config(tmp_path, load_model=load_model))
    assert config.refill is expected
    assert "load_model" not in config.to_json()


def test_refill_must_be_boolean(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="config.refill must be a boolean"):
        load_config(write_config(tmp_path, refill="false"))


def test_refill_and_legacy_load_model_cannot_both_be_set(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cannot both be set"):
        load_config(write_config(tmp_path, refill=False, load_model="fire-once"))


def test_coral_turn_controls_round_trip_and_define_workload_identity(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            framework="coral",
            coral_agent_turns=3,
            coral_global_turns=8,
            coral_restart_exited=True,
        )
    )

    assert config.coral_agent_turns == 3
    assert config.coral_global_turns == 8
    assert config.coral_restart_exited is True
    assert config.to_json()["coral_global_turns"] == 8
    assert config.to_json()["coral_team_size"] == 4
    assert config.workload_identity() == {
        "framework": "coral",
        "concurrency": 8,
        "duration_s": 60,
        "seed": 42,
        "refill": True,
        "concurrency_unit": "coral-team",
        "coral_team_size": 4,
        "coral_dataset": "frontier_cs_algo",
        "coral_agent_turns": 3,
        "coral_global_turns": 8,
        "coral_restart_exited": True,
    }


def test_formal_defaults_are_three_minutes_and_restarting_coral(tmp_path: Path) -> None:
    path = write_config(tmp_path, framework="coral")
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("duration_s")
    path.write_text(json.dumps(value), encoding="utf-8")

    config = load_config(path)

    assert config.duration_s == 180
    assert config.coral_agent_turns == 100
    assert config.coral_global_turns == 10
    assert config.coral_restart_exited is True


def test_coral_team_size_is_fixed_at_four(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="coral_team_size must be 4"):
        load_config(
            write_config(
                tmp_path,
                framework="coral",
                coral_team_size=3,
            )
        )


@pytest.mark.parametrize("value", (1, 3))
def test_coral_global_turns_cannot_be_smaller_than_initial_team(
    tmp_path: Path, value: int
) -> None:
    with pytest.raises(ValidationError, match="four initial CORAL agents"):
        load_config(
            write_config(
                tmp_path,
                framework="coral",
                coral_global_turns=value,
            )
        )


def test_serving_gpu_mode_defaults_to_nvidia(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, serving={"configs": ["serving/configs/test.yaml:1"]})
    )
    assert config.serving_spec().gpu_mode == "nvidia"


def test_serving_gpu_mode_allows_explicit_cdi(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            serving={"configs": ["serving/configs/test.yaml:1"], "gpu_mode": "cdi"},
        )
    )
    assert config.serving_spec().gpu_mode == "cdi"


def test_serving_gpu_mode_rejects_unknown_value(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            serving={"configs": ["serving/configs/test.yaml:1"], "gpu_mode": "auto"},
        )
    )
    with pytest.raises(ValidationError, match="gpu_mode"):
        config.serving_spec()
