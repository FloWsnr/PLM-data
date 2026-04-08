"""Tests for seeded config realization."""

import json
from pathlib import Path
from typing import Any

import yaml

from plm_data.core.config import load_config
from plm_data.core.config_realization import realize_simulation_config
from plm_data.core.runner import SimulationRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config_dict(relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((_REPO_ROOT / relative_path).read_text())


def _write_yaml(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return path


def _contains_sampler_spec(value: Any) -> bool:
    if isinstance(value, dict):
        if "sample" in value:
            return True
        return any(_contains_sampler_spec(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sampler_spec(item) for item in value)
    return False


def _homogeneous_neumann_2d(*, top_value: Any = 0.0) -> dict[str, Any]:
    def _entry(value: Any) -> list[dict[str, Any]]:
        return [
            {
                "operator": "neumann",
                "value": {
                    "type": "constant",
                    "params": {"value": value},
                },
            }
        ]

    return {
        "u": {
            "x-": _entry(0.0),
            "x+": _entry(0.0),
            "y-": _entry(0.0),
            "y+": _entry(top_value),
        }
    }


def test_load_config_rejects_domain_sampling_without_opt_in(tmp_path):
    data = _load_config_dict("configs/basic/heat/2d_localized_blob_diffusion.yaml")
    data["domain"]["size"][0] = {"sample": "uniform", "min": 0.9, "max": 1.1}

    path = _write_yaml(tmp_path, "domain_sampling_disabled.yaml", data)

    try:
        load_config(path)
    except ValueError as exc:
        assert "domain sampling is disabled" in str(exc)
    else:
        raise AssertionError("Expected domain sampling without opt-in to fail.")


def test_load_config_accepts_domain_sampling_with_opt_in(tmp_path):
    data = _load_config_dict("configs/basic/heat/2d_localized_blob_diffusion.yaml")
    data["domain"]["allow_sampling"] = True
    data["domain"]["size"][0] = {"sample": "uniform", "min": 0.9, "max": 1.1}

    path = _write_yaml(tmp_path, "domain_sampling_enabled.yaml", data)
    cfg = load_config(path)

    assert cfg.domain.allow_sampling is True
    assert cfg.domain.params["size"][0]["sample"] == "uniform"


def test_realize_simulation_config_is_seeded_and_concrete(tmp_path):
    data = _load_config_dict(
        "configs/biology/fisher_kpp/2d_logistic_invasion_front.yaml"
    )
    data["parameters"]["D"] = {"sample": "uniform", "min": 0.08, "max": 0.12}
    data["domain"]["allow_sampling"] = True
    data["domain"]["size"][0] = {"sample": "uniform", "min": 8.0, "max": 12.0}
    data["boundary_conditions"] = _homogeneous_neumann_2d(
        top_value={"sample": "uniform", "min": 0.0, "max": 0.2}
    )

    path = _write_yaml(tmp_path, "realize_fisher_kpp.yaml", data)
    cfg = load_config(path)

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    data["seed"] = 7
    cfg_other_seed = load_config(
        _write_yaml(tmp_path, "realize_fisher_kpp_2.yaml", data)
    )
    realized_c = realize_simulation_config(cfg_other_seed)

    assert realized_a.parameters["D"] == realized_b.parameters["D"]
    assert realized_a.domain.params["size"][0] == realized_b.domain.params["size"][0]
    assert (
        realized_a.boundary_field("u").side_conditions("y+")[0].value.params["value"]
        == realized_b.boundary_field("u").side_conditions("y+")[0].value.params["value"]
    )

    assert realized_a.parameters["D"] != realized_c.parameters["D"]
    assert realized_a.domain.params["size"][0] != realized_c.domain.params["size"][0]
    assert (
        realized_a.boundary_field("u").side_conditions("y+")[0].value.params["value"]
        != realized_c.boundary_field("u").side_conditions("y+")[0].value.params["value"]
    )

    assert isinstance(realized_a.parameters["D"], float)
    assert isinstance(realized_a.domain.params["size"][0], float)
    assert isinstance(
        realized_a.boundary_field("u").side_conditions("y+")[0].value.params["value"],
        float,
    )
    assert isinstance(
        realized_a.input("u").initial_condition.params["x_split"],
        float,
    )


def test_realize_simulation_config_concretizes_blob_generators():
    cfg = load_config("configs/basic/heat/2d_localized_blob_diffusion.yaml")

    realized = realize_simulation_config(cfg)

    params = realized.input("u").initial_condition.params
    assert "blobs" in params
    assert "generators" not in params
    assert all("direction" in blob for blob in params["blobs"])


def test_simulation_runner_serializes_realized_config(tmp_path):
    data = _load_config_dict("configs/basic/heat/2d_localized_blob_diffusion.yaml")
    data["domain"]["allow_sampling"] = True
    data["domain"]["size"][0] = {"sample": "uniform", "min": 0.95, "max": 1.05}
    data["coefficients"]["kappa"]["params"]["value"] = {
        "sample": "uniform",
        "min": 0.008,
        "max": 0.012,
    }
    data["boundary_conditions"] = _homogeneous_neumann_2d(
        top_value={"sample": "uniform", "min": 0.0, "max": 0.03}
    )
    data["time"] = {"dt": 0.01, "t_end": 0.02}
    data["output"]["resolution"] = [16, 16]
    data["output"]["num_frames"] = 2

    path = _write_yaml(tmp_path, "sampled_heat.yaml", data)
    runner = SimulationRunner.from_yaml(path, tmp_path / "output")
    summary = runner.run(console_level=50)

    run_meta = json.loads((Path(summary["output_dir"]) / "run_meta.json").read_text())
    resolved = run_meta["config"]["resolved"]

    assert isinstance(resolved["coefficients"]["kappa"]["params"]["value"], float)
    assert isinstance(resolved["domain"]["params"]["size"][0], float)
    assert isinstance(
        resolved["boundary_conditions"]["u"]["sides"]["y+"][0]["value"]["params"][
            "value"
        ],
        float,
    )
    assert "blobs" in resolved["inputs"]["u"]["initial_condition"]["params"]
    assert _contains_sampler_spec(resolved) is False
