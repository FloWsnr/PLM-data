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
        _write_yaml(tmp_path, "realize_fisher_kpp_2.yaml", data),
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


def test_realize_simulation_config_concretizes_annulus_center(tmp_path):
    data = _load_config_dict("configs/basic/heat/2d_annulus_inner_heating_layer.yaml")
    data["domain"]["center"] = [
        {"sample": "uniform", "min": -0.2, "max": 0.2},
        {"sample": "uniform", "min": -0.1, "max": 0.1},
    ]

    cfg = load_config(_write_yaml(tmp_path, "sampled_annulus_center.yaml", data))

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params["center"] == realized_b.domain.params["center"]
    assert all(isinstance(value, float) for value in realized_a.domain.params["center"])


def test_realize_simulation_config_concretizes_basic_annulus_center_refs():
    cfg = load_config("configs/basic/plate/2d_annulus_radial_ringdown.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params["center"] == realized_b.domain.params["center"]
    assert (
        realized_a.coefficient("rigidity").params["center"]
        == realized_a.domain.params["center"]
    )
    assert (
        realized_a.input("deflection").initial_condition.params["center"]
        == (realized_a.domain.params["center"])
    )


def test_realize_simulation_config_concretizes_y_bifurcation_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_y_bifurcation_split_diffusion.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["inlet_length"], float)
    assert isinstance(realized_a.domain.params["branch_length"], float)
    assert isinstance(realized_a.domain.params["branch_angle_degrees"], float)
    assert isinstance(realized_a.domain.params["channel_width"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 0.9 <= realized_a.domain.params["inlet_length"] <= 1.08
    assert 0.82 <= realized_a.domain.params["branch_length"] <= 0.98
    assert 34.0 <= realized_a.domain.params["branch_angle_degrees"] <= 42.0
    assert 0.2 <= realized_a.domain.params["channel_width"] <= 0.26
    assert 0.028 <= realized_a.domain.params["mesh_size"] <= 0.036


def test_realize_simulation_config_concretizes_venturi_channel_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_venturi_channel_throat_focusing.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["length"], float)
    assert isinstance(realized_a.domain.params["height"], float)
    assert isinstance(realized_a.domain.params["throat_height"], float)
    assert isinstance(realized_a.domain.params["constriction_center_x"], float)
    assert isinstance(realized_a.domain.params["constriction_radius"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 2.0 <= realized_a.domain.params["length"] <= 2.3
    assert 0.9 <= realized_a.domain.params["height"] <= 1.1
    assert 0.34 <= realized_a.domain.params["throat_height"] <= 0.5
    assert 0.95 <= realized_a.domain.params["constriction_center_x"] <= 1.15
    assert 0.4 <= realized_a.domain.params["constriction_radius"] <= 0.65
    assert 0.028 <= realized_a.domain.params["mesh_size"] <= 0.036


def test_realize_simulation_config_concretizes_serpentine_channel_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_serpentine_channel_guided_diffusion.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["channel_length"], float)
    assert isinstance(realized_a.domain.params["lane_spacing"], float)
    assert isinstance(realized_a.domain.params["n_bends"], int)
    assert isinstance(realized_a.domain.params["channel_width"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 0.88 <= realized_a.domain.params["channel_length"] <= 1.08
    assert 0.36 <= realized_a.domain.params["lane_spacing"] <= 0.48
    assert 2 <= realized_a.domain.params["n_bends"] <= 4
    assert 0.16 <= realized_a.domain.params["channel_width"] <= 0.22
    assert 0.026 <= realized_a.domain.params["mesh_size"] <= 0.034


def test_realize_simulation_config_concretizes_l_shape_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_l_shape_corner_heating.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["outer_width"], float)
    assert isinstance(realized_a.domain.params["outer_height"], float)
    assert isinstance(realized_a.domain.params["cutout_width"], float)
    assert isinstance(realized_a.domain.params["cutout_height"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 0.96 <= realized_a.domain.params["outer_width"] <= 1.12
    assert 0.96 <= realized_a.domain.params["outer_height"] <= 1.12
    assert 0.34 <= realized_a.domain.params["cutout_width"] <= 0.46
    assert 0.34 <= realized_a.domain.params["cutout_height"] <= 0.46
    assert 0.024 <= realized_a.domain.params["mesh_size"] <= 0.032


def test_realize_simulation_config_concretizes_airfoil_channel_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_airfoil_channel_heat_shadow.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["length"], float)
    assert isinstance(realized_a.domain.params["height"], float)
    assert all(
        isinstance(value, float) for value in realized_a.domain.params["airfoil_center"]
    )
    assert isinstance(realized_a.domain.params["chord_length"], float)
    assert isinstance(realized_a.domain.params["thickness_ratio"], float)
    assert isinstance(realized_a.domain.params["attack_angle_degrees"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 2.4 <= realized_a.domain.params["length"] <= 2.8
    assert 1.0 <= realized_a.domain.params["height"] <= 1.15
    assert 1.05 <= realized_a.domain.params["airfoil_center"][0] <= 1.35
    assert 0.48 <= realized_a.domain.params["airfoil_center"][1] <= 0.66
    assert 0.52 <= realized_a.domain.params["chord_length"] <= 0.64
    assert 0.1 <= realized_a.domain.params["thickness_ratio"] <= 0.16
    assert -10.0 <= realized_a.domain.params["attack_angle_degrees"] <= 10.0
    assert 0.028 <= realized_a.domain.params["mesh_size"] <= 0.036


def test_realize_simulation_config_concretizes_multi_hole_plate_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_multi_hole_plate_hole_heating.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["width"], float)
    assert isinstance(realized_a.domain.params["height"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert len(realized_a.domain.params["holes"]) == 3
    for hole in realized_a.domain.params["holes"]:
        assert isinstance(hole["center"][0], float)
        assert isinstance(hole["center"][1], float)
        assert isinstance(hole["radius"], float)
    assert 0.024 <= realized_a.domain.params["mesh_size"] <= 0.032


def test_realize_simulation_config_concretizes_side_cavity_channel_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_side_cavity_channel_delayed_release.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["length"], float)
    assert isinstance(realized_a.domain.params["height"], float)
    assert isinstance(realized_a.domain.params["cavity_width"], float)
    assert isinstance(realized_a.domain.params["cavity_depth"], float)
    assert isinstance(realized_a.domain.params["cavity_center_x"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 2.2 <= realized_a.domain.params["length"] <= 2.8
    assert 0.68 <= realized_a.domain.params["height"] <= 0.82
    assert 0.36 <= realized_a.domain.params["cavity_width"] <= 0.56
    assert 0.28 <= realized_a.domain.params["cavity_depth"] <= 0.48
    assert 1.0 <= realized_a.domain.params["cavity_center_x"] <= 1.55
    assert 0.026 <= realized_a.domain.params["mesh_size"] <= 0.034


def test_realize_simulation_config_concretizes_porous_channel_domain_sampling():
    cfg = load_config("configs/basic/heat/2d_porous_channel_trapping_diffusion.yaml")

    realized_a = realize_simulation_config(cfg)
    realized_b = realize_simulation_config(cfg)

    assert realized_a.domain.params == realized_b.domain.params
    assert isinstance(realized_a.domain.params["length"], float)
    assert isinstance(realized_a.domain.params["height"], float)
    assert isinstance(realized_a.domain.params["obstacle_radius"], float)
    assert isinstance(realized_a.domain.params["n_rows"], int)
    assert isinstance(realized_a.domain.params["n_cols"], int)
    assert isinstance(realized_a.domain.params["pitch_x"], float)
    assert isinstance(realized_a.domain.params["pitch_y"], float)
    assert isinstance(realized_a.domain.params["x_margin"], float)
    assert isinstance(realized_a.domain.params["y_margin"], float)
    assert isinstance(realized_a.domain.params["row_shift_fraction"], float)
    assert isinstance(realized_a.domain.params["mesh_size"], float)
    assert 2.55 <= realized_a.domain.params["length"] <= 2.75
    assert realized_a.domain.params["height"] == 1.0
    assert 0.085 <= realized_a.domain.params["obstacle_radius"] <= 0.11
    assert 2 <= realized_a.domain.params["n_rows"] <= 3
    assert 4 <= realized_a.domain.params["n_cols"] <= 5
    assert 0.34 <= realized_a.domain.params["pitch_x"] <= 0.4
    assert 0.26 <= realized_a.domain.params["pitch_y"] <= 0.28
    assert 0.2 <= realized_a.domain.params["x_margin"] <= 0.28
    assert 0.12 <= realized_a.domain.params["y_margin"] <= 0.16
    assert 0.32 <= realized_a.domain.params["row_shift_fraction"] <= 0.48
    assert 0.026 <= realized_a.domain.params["mesh_size"] <= 0.034


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
