"""Tests for output management."""

import json
import numpy as np
import pytest
from pathlib import Path
from pde import ScalarField, FieldCollection, CartesianGrid

from pde_sim.core.output import OutputManager, create_metadata
from pde_sim.core.config import SimulationConfig, OutputConfig, BoundaryConfig, InitialConditionConfig
from pde_sim.pdes.base import PDEMetadata, PDEParameter


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_initialization(self, tmp_output):
        """Test OutputManager initialization."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-sim-123",
            colormap="viridis",
        )

        assert manager.sim_id == "test-sim-123"
        assert manager.colormap == "viridis"
        assert manager.output_dir == tmp_output / "test-sim-123"
        assert manager.frames_dir.exists()

    def test_save_scalar_frame(self, tmp_output, small_grid):
        """Test saving a scalar field frame."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-scalar",
        )

        field = ScalarField.random_uniform(small_grid)
        frame_path = manager.save_frame(field, frame_index=0, simulation_time=0.0)

        assert frame_path.exists()
        assert frame_path.name == "000000.png"

    def test_save_multiple_frames(self, tmp_output, small_grid):
        """Test saving multiple frames."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-multi",
        )

        field = ScalarField.random_uniform(small_grid)

        for i in range(5):
            manager.save_frame(field, frame_index=i, simulation_time=i * 0.1)

        # Check all frames exist
        frames = list(manager.frames_dir.glob("*.png"))
        assert len(frames) == 5

        # Check frame annotations
        annotations = manager.get_frame_annotations()
        assert len(annotations) == 5
        assert annotations[0]["frameIndex"] == 0
        assert annotations[4]["frameIndex"] == 4

    def test_save_field_collection_frame(self, tmp_output, small_grid):
        """Test saving a FieldCollection frame."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-collection",
            field_to_plot="u",
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        frame_path = manager.save_frame(collection, frame_index=0, simulation_time=0.0)
        assert frame_path.exists()

    def test_value_range_tracking(self, tmp_output, small_grid):
        """Test that min/max values are tracked across frames."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-range",
        )

        # First frame with known values
        data1 = np.zeros(small_grid.shape)
        data1[0, 0] = 0.0
        data1[1, 1] = 1.0
        field1 = ScalarField(small_grid, data1)
        manager.save_frame(field1, 0, 0.0)

        # Second frame with a wider range
        data2 = np.zeros(small_grid.shape)
        data2[0, 0] = -2.0
        data2[1, 1] = 3.0
        field2 = ScalarField(small_grid, data2)
        manager.save_frame(field2, 1, 0.1)

        # vmin should be -2, vmax should be 3
        assert manager.vmin == -2.0
        assert manager.vmax == 3.0

    def test_save_metadata(self, tmp_output, small_grid):
        """Test saving metadata JSON."""
        manager = OutputManager(
            base_path=tmp_output,
            sim_id="test-meta",
        )

        field = ScalarField.random_uniform(small_grid)
        manager.save_frame(field, 0, 0.0)

        metadata = {"test": "value", "visualization": {}}
        meta_path = manager.save_metadata(metadata)

        assert meta_path.exists()
        with open(meta_path) as f:
            saved = json.load(f)
        assert saved["test"] == "value"
        assert "minValue" in saved["visualization"]


class TestCreateMetadata:
    """Tests for create_metadata function."""

    def test_create_complete_metadata(self):
        """Test creating a complete metadata dictionary."""
        preset_meta = PDEMetadata(
            name="test-pde",
            category="test",
            description="Test PDE",
            equations={"u": "laplace(u)"},
            parameters=[PDEParameter("D", 1.0, "Diffusion")],
            num_fields=1,
            field_names=["u"],
        )

        config = SimulationConfig(
            preset="test-pde",
            parameters={"D": 1.0},
            init=InitialConditionConfig(type="random", params={}),
            solver="euler",
            timesteps=100,
            dt=0.01,
            resolution=64,
            bc=BoundaryConfig(x="periodic", y="periodic"),
            output=OutputConfig(path=Path("./output"), colormap="turbo"),
            seed=42,
            domain_size=1.0,
        )

        frame_annotations = [
            {"frameIndex": 0, "simulationTime": 0.0},
            {"frameIndex": 1, "simulationTime": 0.1},
        ]

        metadata = create_metadata(
            sim_id="test-123",
            preset_name="test-pde",
            preset_metadata=preset_meta,
            config=config,
            total_time=0.1,
            frame_annotations=frame_annotations,
        )

        # Check required fields
        assert metadata["id"] == "test-123"
        assert metadata["preset"] == "test-pde"
        assert "timestamp" in metadata
        assert "generatorVersion" in metadata

        # Check equations
        assert metadata["equations"]["u"] == "laplace(u)"
        assert metadata["boundaryConditions"] == {"x": "periodic", "y": "periodic"}
        assert metadata["initialConditions"] == "random"

        # Check parameters
        assert metadata["parameters"]["kinetic"]["D"] == 1.0
        assert metadata["parameters"]["dt"] == 0.01
        assert metadata["parameters"]["numSpecies"] == 1

        # Check simulation info
        assert metadata["simulation"]["totalFrames"] == 2
        assert metadata["simulation"]["resolution"] == [64, 64]

        # Check frame annotations
        assert len(metadata["frameAnnotations"]) == 2

        # Check description field exists (may be None for non-existent presets)
        assert "description" in metadata

    def test_create_metadata_with_description(self):
        """Test that description is loaded from markdown files for real presets."""
        preset_meta = PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott model",
            equations={"u": "D_u*laplace(u) - u*v**2 + F*(1-u)", "v": "D_v*laplace(v) + u*v**2 - (F+k)*v"},
            parameters=[PDEParameter("D_u", 0.2, "U diffusion"), PDEParameter("D_v", 0.1, "V diffusion")],
            num_fields=2,
            field_names=["u", "v"],
        )

        config = SimulationConfig(
            preset="gray-scott",
            parameters={"D_u": 0.2, "D_v": 0.1, "F": 0.04, "k": 0.06},
            init=InitialConditionConfig(type="random", params={}),
            solver="euler",
            timesteps=100,
            dt=0.01,
            resolution=64,
            bc=BoundaryConfig(x="periodic", y="periodic"),
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

        # Description should be loaded from gray-scott.md
        assert metadata["description"] is not None
        assert "Gray-Scott" in metadata["description"]
        assert "## Mathematical Formulation" in metadata["description"]

    def test_create_metadata_description_not_found(self):
        """Test that description is None for presets without markdown files."""
        preset_meta = PDEMetadata(
            name="nonexistent-pde",
            category="test",
            description="Nonexistent PDE",
            equations={"u": "laplace(u)"},
            parameters=[],
            num_fields=1,
            field_names=["u"],
        )

        config = SimulationConfig(
            preset="nonexistent-pde",
            parameters={},
            init=InitialConditionConfig(type="random", params={}),
            solver="euler",
            timesteps=100,
            dt=0.01,
            resolution=64,
            bc=BoundaryConfig(x="periodic", y="periodic"),
            output=OutputConfig(path=Path("./output"), colormap="turbo"),
            seed=42,
            domain_size=1.0,
        )

        metadata = create_metadata(
            sim_id="test-nonexistent",
            preset_name="nonexistent-pde",
            preset_metadata=preset_meta,
            config=config,
            total_time=1.0,
            frame_annotations=[],
        )

        # Description should be None since no markdown file exists
        assert metadata["description"] is None
