"""Tests for output management."""

import json
from pathlib import Path

import numpy as np
from pde import ScalarField, FieldCollection

from pde_sim.core.output import OutputManager, create_metadata
from pde_sim.core.config import SimulationConfig, OutputConfig, BoundaryConfig, InitialConditionConfig
from pde_sim.pdes.base import PDEMetadata, PDEParameter


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_initialization(self, tmp_output):
        """Test OutputManager initialization."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="heat_2024-01-15",
            colormap="viridis",
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
        )

        field = ScalarField.random_uniform(small_grid)
        frame_path = manager.save_frame(field, frame_index=0, simulation_time=0.0)

        assert frame_path.exists()
        assert frame_path.name == "000000.png"

    def test_save_multiple_frames(self, tmp_output, small_grid):
        """Test saving multiple frames."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-multi_2024-01-15",
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
            folder_name="test-collection_2024-01-15",
            field_to_plot="u",
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        frame_path = manager.save_frame(collection, frame_index=0, simulation_time=0.0)
        assert frame_path.exists()

    def test_compute_range(self, tmp_output, small_grid):
        """Test that compute_range pre-computes global min/max."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-compute-range_2024-01-15",
        )

        # Create frames with different ranges
        data1 = np.zeros(small_grid.shape)
        data1[0, 0] = 0.0
        data1[1, 1] = 1.0
        field1 = ScalarField(small_grid, data1)

        data2 = np.zeros(small_grid.shape)
        data2[0, 0] = -2.0
        data2[1, 1] = 3.0
        field2 = ScalarField(small_grid, data2)

        # Pre-compute range from all frames
        vmin, vmax = manager.compute_range([field1, field2])

        assert vmin == -2.0
        assert vmax == 3.0
        assert manager.vmin == -2.0
        assert manager.vmax == 3.0

    def test_consistent_colorscale_with_compute_range(self, tmp_output, small_grid):
        """Test that all frames use consistent colorscale when using compute_range."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-consistent_2024-01-15",
        )

        # Create frames with different value ranges
        data1 = np.ones(small_grid.shape) * 0.5  # range: [0.5, 0.5]
        field1 = ScalarField(small_grid, data1)

        data2 = np.zeros(small_grid.shape)
        data2[0, 0] = -2.0
        data2[1, 1] = 3.0  # range: [-2.0, 3.0]
        field2 = ScalarField(small_grid, data2)

        # Two-pass approach: compute range first, then save frames
        manager.compute_range([field1, field2])

        # Both frames should use the global range [-2.0, 3.0]
        manager.save_frame(field1, 0, 0.0)
        manager.save_frame(field2, 1, 0.1)

        # Verify the range remained consistent
        assert manager.vmin == -2.0
        assert manager.vmax == 3.0

        # Both frames were saved
        frames = list(manager.frames_dir.glob("*.png"))
        assert len(frames) == 2

    def test_save_metadata(self, tmp_output, small_grid):
        """Test saving metadata JSON."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-meta_2024-01-15",
        )

        field = ScalarField.random_uniform(small_grid)
        manager.save_frame(field, 0, 0.0)

        metadata = {"test": "value", "visualization": {}}
        meta_path = manager.save_metadata(metadata)

        assert meta_path.exists()
        with open(meta_path) as f:
            saved = json.load(f)
        assert saved["test"] == "value"
        assert "colorbarMin" in saved["visualization"]
        assert "colorbarMax" in saved["visualization"]

    def test_save_trajectory_array_scalar(self, tmp_output, small_grid):
        """Test saving trajectory as numpy array for scalar fields."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-trajectory_2024-01-15",
            save_array=True,
        )

        # Create a sequence of scalar fields
        states = [ScalarField.random_uniform(small_grid) for _ in range(5)]
        times = [0.0, 0.1, 0.2, 0.3, 0.4]

        array_path = manager.save_trajectory_array(states, times)

        # Check file exists
        assert array_path.exists()
        assert array_path.name == "trajectory.npz"

        # Load and verify contents
        data = np.load(array_path)
        assert "trajectory" in data
        assert "times" in data

        # Check shapes
        assert data["trajectory"].shape == (5, 32, 32)
        assert data["times"].shape == (5,)

        # Check times values
        np.testing.assert_array_almost_equal(data["times"], times)

    def test_save_trajectory_array_field_collection(self, tmp_output, small_grid):
        """Test saving trajectory as numpy array for multi-field systems."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-trajectory-collection_2024-01-15",
            field_to_plot="u",
            save_array=True,
        )

        # Create a sequence of field collections
        states = []
        for _ in range(3):
            u = ScalarField.random_uniform(small_grid)
            u.label = "u"
            v = ScalarField.random_uniform(small_grid)
            v.label = "v"
            states.append(FieldCollection([u, v]))

        times = [0.0, 0.5, 1.0]

        array_path = manager.save_trajectory_array(states, times)

        # Load and verify
        data = np.load(array_path)

        # Should only save the selected field (u)
        assert data["trajectory"].shape == (3, 32, 32)
        assert data["times"].shape == (3,)


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
            t_end=100,
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
            t_end=100,
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
        assert "## Equations" in metadata["description"]

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
            t_end=100,
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


class TestMagnitudePlotting:
    """Tests for magnitude and vector plotting functionality."""

    def test_parse_magnitude_syntax(self, tmp_output):
        """Test parsing mag(X,Y) syntax."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-mag-parse_2024-01-15",
            field_to_plot="mag(X,Y)",
        )

        assert manager.plot_mode == "magnitude"
        assert manager.plot_fields == ["X", "Y"]

    def test_parse_magnitude_with_spaces(self, tmp_output):
        """Test parsing mag(X, Y) with space after comma."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-mag-space_2024-01-15",
            field_to_plot="mag(u, v)",
        )

        assert manager.plot_mode == "magnitude"
        assert manager.plot_fields == ["u", "v"]

    def test_parse_single_field(self, tmp_output):
        """Test parsing single field name."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-single-field_2024-01-15",
            field_to_plot="u",
        )

        assert manager.plot_mode == "single"
        assert manager.plot_fields == ["u"]

    def test_parse_no_field(self, tmp_output):
        """Test parsing when no field is specified."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-no-field_2024-01-15",
        )

        assert manager.plot_mode == "single"
        assert manager.plot_fields == []

    def test_extract_magnitude_data(self, tmp_output, small_grid):
        """Test extracting magnitude from two fields."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-mag-data_2024-01-15",
            field_to_plot="mag(X,Y)",
        )

        # Create field collection with known values
        x_data = np.ones(small_grid.shape) * 3.0
        y_data = np.ones(small_grid.shape) * 4.0

        X = ScalarField(small_grid, x_data)
        X.label = "X"
        Y = ScalarField(small_grid, y_data)
        Y.label = "Y"
        collection = FieldCollection([X, Y])

        # Extract magnitude
        data = manager._extract_field_data(collection)

        # Should be sqrt(3^2 + 4^2) = 5
        np.testing.assert_array_almost_equal(data, np.ones(small_grid.shape) * 5.0)

    def test_extract_vector_components(self, tmp_output, small_grid):
        """Test extracting vector components for quiver."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-vector-comp_2024-01-15",
            field_to_plot="mag(u,v)",
            show_vectors=True,
        )

        u_data = np.random.rand(*small_grid.shape)
        v_data = np.random.rand(*small_grid.shape)

        u = ScalarField(small_grid, u_data)
        u.label = "u"
        v = ScalarField(small_grid, v_data)
        v.label = "v"
        collection = FieldCollection([u, v])

        components = manager._extract_vector_components(collection)

        assert components is not None
        np.testing.assert_array_equal(components[0], u_data)
        np.testing.assert_array_equal(components[1], v_data)

    def test_extract_vector_components_single_mode(self, tmp_output, small_grid):
        """Test that vector extraction returns None for single field mode."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-vector-single_2024-01-15",
            field_to_plot="u",
        )

        u = ScalarField.random_uniform(small_grid)
        u.label = "u"
        v = ScalarField.random_uniform(small_grid)
        v.label = "v"
        collection = FieldCollection([u, v])

        components = manager._extract_vector_components(collection)
        assert components is None

    def test_save_frame_with_vectors(self, tmp_output, small_grid):
        """Test saving frame with vector overlay."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-vectors_2024-01-15",
            field_to_plot="mag(X,Y)",
            show_vectors=True,
            vector_density=8,
        )

        # Create field collection
        X = ScalarField.random_uniform(small_grid)
        X.label = "X"
        Y = ScalarField.random_uniform(small_grid)
        Y.label = "Y"
        collection = FieldCollection([X, Y])

        # Save frame
        frame_path = manager.save_frame(collection, frame_index=0, simulation_time=0.0)

        assert frame_path.exists()
        assert frame_path.name == "000000.png"

    def test_compute_range_with_magnitude(self, tmp_output, small_grid):
        """Test computing range for magnitude mode."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-mag-range_2024-01-15",
            field_to_plot="mag(X,Y)",
        )

        # Create field collections with known values
        x1 = np.ones(small_grid.shape) * 3.0
        y1 = np.ones(small_grid.shape) * 4.0  # mag = 5
        X1 = ScalarField(small_grid, x1)
        X1.label = "X"
        Y1 = ScalarField(small_grid, y1)
        Y1.label = "Y"
        state1 = FieldCollection([X1, Y1])

        x2 = np.ones(small_grid.shape) * 6.0
        y2 = np.ones(small_grid.shape) * 8.0  # mag = 10
        X2 = ScalarField(small_grid, x2)
        X2.label = "X"
        Y2 = ScalarField(small_grid, y2)
        Y2.label = "Y"
        state2 = FieldCollection([X2, Y2])

        vmin, vmax = manager.compute_range([state1, state2])

        assert vmin == 5.0
        assert vmax == 10.0
