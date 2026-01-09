"""Tests for output management."""

import json
from pathlib import Path

import numpy as np
import pytest
from pde import ScalarField, FieldCollection, CartesianGrid

from pde_sim.core.output import OutputManager, create_metadata
from pde_sim.core.config import (
    SimulationConfig,
    OutputConfig,
    BoundaryConfig,
    InitialConditionConfig,
)
from pde_sim.pdes.base import PDEMetadata, PDEParameter


@pytest.fixture
def tmp_output(tmp_path):
    """Create a temporary output directory."""
    return tmp_path


@pytest.fixture
def small_grid():
    """Create a small grid for testing."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16])


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_initialization(self, tmp_output):
        """Test OutputManager initialization."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="heat_2024-01-15",
            colormap="viridis",
            field_configs=[("u", "viridis")],
        )

        assert manager.folder_name == "heat_2024-01-15"
        assert manager.colormap == "viridis"
        assert manager.output_dir == tmp_output / "heat_2024-01-15"
        assert manager.frames_dir.exists()

    def test_save_scalar_frame(self, tmp_output, small_grid):
        """Test saving a scalar field frame."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-scalar_2024-01-15",
            field_configs=[("u", "turbo")],
        )

        field = ScalarField.random_uniform(small_grid)
        field.label = "u"

        manager.compute_range_for_field([field], "u")
        frame_path = manager.save_frame(field, "u", frame_index=0, simulation_time=0.0)

        assert frame_path.exists()
        assert frame_path.name == "u_000000.png"

    def test_save_multiple_frames(self, tmp_output, small_grid):
        """Test saving multiple frames."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-multi_2024-01-15",
            field_configs=[("u", "turbo")],
        )

        field = ScalarField.random_uniform(small_grid)
        field.label = "u"

        manager.compute_range_for_field([field], "u")

        for i in range(3):
            manager.save_frame(field, "u", frame_index=i, simulation_time=i * 0.1)

        frames = list(manager.frames_dir.glob("*.png"))
        assert len(frames) == 3

    def test_save_field_collection_frame(self, tmp_output, small_grid):
        """Test saving a frame from a FieldCollection."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-collection_2024-01-15",
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        manager.compute_range_for_field([collection], "u")
        manager.compute_range_for_field([collection], "v")

        paths = manager.save_all_fields(collection, frame_index=0, simulation_time=0.0)

        assert len(paths) == 2
        assert (manager.frames_dir / "u_000000.png").exists()
        assert (manager.frames_dir / "v_000000.png").exists()

    def test_compute_range_for_field(self, tmp_output, small_grid):
        """Test computing range for a specific field."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-range_2024-01-15",
            field_configs=[("u", "viridis")],
        )

        # Create fields with known min/max
        field1 = ScalarField(small_grid, np.ones(small_grid.shape) * 0.5)
        field1.label = "u"
        field2 = ScalarField(small_grid, np.ones(small_grid.shape) * 1.5)
        field2.label = "u"

        vmin, vmax = manager.compute_range_for_field([field1, field2], "u")

        assert vmin == 0.5
        assert vmax == 1.5
        assert manager.field_ranges["u"] == (0.5, 1.5)

    def test_save_metadata(self, tmp_output, small_grid):
        """Test saving metadata."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-meta_2024-01-15",
            field_configs=[("u", "viridis")],
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"

        manager.compute_range_for_field([u], "u")

        metadata = {"test": "data"}
        metadata_path = manager.save_metadata(metadata)

        assert metadata_path.exists()
        with open(metadata_path) as f:
            saved = json.load(f)
        assert saved["test"] == "data"
        assert "visualization" in saved
        assert "fields" in saved["visualization"]
        assert "u" in saved["visualization"]["fields"]

    def test_save_trajectory_array(self, tmp_output, small_grid):
        """Test saving trajectory as numpy array."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-traj_2024-01-15",
            save_array=True,
            field_configs=[("u", "turbo")],
        )

        states = []
        for i in range(5):
            field = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            field.label = "u"
            states.append(field)
        times = [0.0, 0.1, 0.2, 0.3, 0.4]

        array_path = manager.save_trajectory_array(states, times, "u")

        assert array_path.exists()
        assert array_path.name == "trajectory_u.npz"

        # Verify contents
        data = np.load(array_path)
        assert "trajectory" in data
        assert "times" in data
        assert data["trajectory"].shape == (5, 16, 16)
        np.testing.assert_array_equal(data["times"], times)

    def test_save_trajectory_array_multi_field(self, tmp_output, small_grid):
        """Test saving trajectory for multiple fields."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-traj-multi_2024-01-15",
            save_array=True,
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        states = []
        for i in range(3):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))
        times = [0.0, 0.1, 0.2]

        # Save trajectory for both fields
        array_path_u = manager.save_trajectory_array(states, times, "u")
        array_path_v = manager.save_trajectory_array(states, times, "v")

        assert array_path_u.exists()
        assert array_path_v.exists()

        data_u = np.load(array_path_u)
        data_v = np.load(array_path_v)
        assert data_u["trajectory"].shape == (3, 16, 16)
        assert data_v["trajectory"].shape == (3, 16, 16)


class TestCreateMetadata:
    """Tests for create_metadata function."""

    def test_create_complete_metadata(self):
        """Test creating complete metadata."""
        preset_meta = PDEMetadata(
            name="test-pde",
            category="test",
            description="Test PDE",
            equations={"u": "laplace(u)"},
            parameters=[PDEParameter("D", 0.1, "diffusion")],
            num_fields=1,
            field_names=["u"],
        )

        config = SimulationConfig(
            preset="test-pde",
            parameters={"D": 0.1},
            init=InitialConditionConfig(type="random", params={}),
            solver="euler",
            t_end=100,
            dt=0.01,
            resolution=64,
            bc=BoundaryConfig(),
            output=OutputConfig(path=Path("./output"), colormap="turbo",
                                fields=["u:turbo"]),
            seed=42,
            domain_size=1.0,
        )

        frame_annotations = [
            {"frameIndex": 0, "simulationTime": 0.0},
            {"frameIndex": 1, "simulationTime": 0.5},
        ]

        metadata = create_metadata(
            sim_id="test-123",
            preset_name="test-pde",
            preset_metadata=preset_meta,
            config=config,
            total_time=1.0,
            frame_annotations=frame_annotations,
        )

        assert metadata["id"] == "test-123"
        assert metadata["preset"] == "test-pde"
        assert metadata["equations"] == {"u": "laplace(u)"}
        assert metadata["simulation"]["resolution"] == [64, 64]
        assert len(metadata["frameAnnotations"]) == 2

    def test_create_metadata_with_description(self):
        """Test that description is loaded for real presets."""
        preset_meta = PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott model",
            equations={"u": "D_u*laplace(u)", "v": "D_v*laplace(v)"},
            parameters=[PDEParameter("D_u", 0.2, "U diffusion")],
            num_fields=2,
            field_names=["u", "v"],
        )

        config = SimulationConfig(
            preset="gray-scott",
            parameters={"D_u": 0.2},
            init=InitialConditionConfig(type="random", params={}),
            solver="euler",
            t_end=100,
            dt=0.01,
            resolution=64,
            bc=BoundaryConfig(),
            output=OutputConfig(path=Path("./output"), colormap="turbo"),
            seed=42,
            domain_size=1.0,
        )

        metadata = create_metadata(
            sim_id="test-gray-scott",
            preset_name="gray-scott",
            preset_metadata=preset_meta,
            config=config,
            total_time=1.0,
            frame_annotations=[],
        )

        assert metadata["description"] is not None
        assert "Gray-Scott" in metadata["description"]


class TestMultiFieldOutput:
    """Tests for multi-field output functionality."""

    def test_initialization_with_field_configs(self, tmp_output):
        """Test OutputManager initialization with field configs."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-multi_2024-01-15",
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        assert manager.field_configs == [("u", "viridis"), ("v", "plasma")]
        assert manager.field_colormaps == {"u": "viridis", "v": "plasma"}

    def test_compute_range_per_field(self, tmp_output, small_grid):
        """Test computing range for each field separately."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-range-field_2024-01-15",
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        u1 = ScalarField(small_grid, np.ones(small_grid.shape) * 0.5)
        u1.label = "u"
        v1 = ScalarField(small_grid, np.ones(small_grid.shape) * 1.0)
        v1.label = "v"
        state1 = FieldCollection([u1, v1])

        u2 = ScalarField(small_grid, np.ones(small_grid.shape) * 1.5)
        u2.label = "u"
        v2 = ScalarField(small_grid, np.ones(small_grid.shape) * 2.0)
        v2.label = "v"
        state2 = FieldCollection([u2, v2])

        vmin_u, vmax_u = manager.compute_range_for_field([state1, state2], "u")
        vmin_v, vmax_v = manager.compute_range_for_field([state1, state2], "v")

        assert vmin_u == 0.5
        assert vmax_u == 1.5
        assert vmin_v == 1.0
        assert vmax_v == 2.0

    def test_save_all_fields(self, tmp_output, small_grid):
        """Test saving frames for all configured fields."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-save-all_2024-01-15",
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        manager.compute_range_for_field([collection], "u")
        manager.compute_range_for_field([collection], "v")

        paths = manager.save_all_fields(collection, 0, 0.0)

        assert len(paths) == 2
        assert (manager.frames_dir / "u_000000.png").exists()
        assert (manager.frames_dir / "v_000000.png").exists()
        assert len(manager.saved_frames) == 1

    def test_filename_format(self, tmp_output, small_grid):
        """Test that filenames follow {field}_{frame:06d}.png format."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-filename_2024-01-15",
            field_configs=[("X", "viridis"), ("Y", "plasma")],
        )

        X = ScalarField.random_uniform(small_grid)
        X.label = "X"
        Y = ScalarField.random_uniform(small_grid)
        Y.label = "Y"
        collection = FieldCollection([X, Y])

        manager.compute_range_for_field([collection], "X")
        manager.compute_range_for_field([collection], "Y")

        for i in range(3):
            manager.save_all_fields(collection, i, i * 0.1)

        expected = [
            "X_000000.png", "Y_000000.png",
            "X_000001.png", "Y_000001.png",
            "X_000002.png", "Y_000002.png",
        ]
        for name in expected:
            assert (manager.frames_dir / name).exists(), f"Missing {name}"

    def test_save_metadata_multi_field(self, tmp_output, small_grid):
        """Test that metadata includes per-field visualization info."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-metadata-multi_2024-01-15",
            field_configs=[("u", "viridis"), ("v", "plasma")],
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        manager.compute_range_for_field([collection], "u")
        manager.compute_range_for_field([collection], "v")

        metadata = {"test": "data"}
        metadata_path = manager.save_metadata(metadata)

        with open(metadata_path) as f:
            saved = json.load(f)

        assert "visualization" in saved
        assert "fields" in saved["visualization"]
        assert "u" in saved["visualization"]["fields"]
        assert "v" in saved["visualization"]["fields"]
        assert saved["visualization"]["fields"]["u"]["colormap"] == "viridis"
        assert saved["visualization"]["fields"]["v"]["colormap"] == "plasma"
