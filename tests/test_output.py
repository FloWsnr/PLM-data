"""Tests for output management."""

import json
from pathlib import Path

import h5py
import imageio
import numpy as np
import pytest
from pde import ScalarField, FieldCollection, CartesianGrid

from pde_sim.core.output import (
    OutputManager,
    create_metadata,
    PNGHandler,
    MP4Handler,
    GIFHandler,
    NumpyHandler,
    H5Handler,
    create_output_handler,
)
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


class TestOutputHandler:
    """Tests for OutputHandler classes."""

    def test_create_png_handler(self):
        """Test creating PNG handler."""
        handler = create_output_handler("png")
        assert isinstance(handler, PNGHandler)

    def test_create_numpy_handler(self):
        """Test creating Numpy handler."""
        handler = create_output_handler("numpy")
        assert isinstance(handler, NumpyHandler)

    def test_create_h5_handler(self):
        """Test creating H5 handler."""
        handler = create_output_handler("h5")
        assert isinstance(handler, H5Handler)

    def test_create_mp4_handler(self):
        """Test creating MP4 handler."""
        handler = create_output_handler("mp4", fps=30)
        assert isinstance(handler, MP4Handler)
        assert handler.fps == 30

    def test_create_gif_handler(self):
        """Test creating GIF handler."""
        handler = create_output_handler("gif", fps=15)
        assert isinstance(handler, GIFHandler)
        assert handler.fps == 15

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unknown output format"):
            create_output_handler("invalid")


class TestPNGHandler:
    """Tests for PNGHandler."""

    def test_creates_field_subdirectories(self, tmp_output):
        """Test PNG handler creates per-field subdirectories."""
        handler = PNGHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=128,
            figsize=(8, 8),
        )

        assert (tmp_output / "frames" / "u").exists()
        assert (tmp_output / "frames" / "v").exists()

    def test_frame_naming(self, tmp_output):
        """Test frames are named {frame:06d}.png in subdirectories."""
        handler = PNGHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=64,
            figsize=(4, 4),
        )

        data = np.random.rand(16, 16)
        handler.save_frame(data, "u", 0, 0.0, 0.0, 1.0, "viridis")
        handler.save_frame(data, "u", 1, 0.1, 0.0, 1.0, "viridis")

        assert (tmp_output / "frames" / "u" / "000000.png").exists()
        assert (tmp_output / "frames" / "u" / "000001.png").exists()

    def test_finalize_metadata(self, tmp_output):
        """Test finalize returns correct metadata."""
        handler = PNGHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=64,
            figsize=(4, 4),
        )

        metadata = handler.finalize()

        assert metadata["format"] == "png"
        assert metadata["framesDirectory"] == "frames/"
        assert "u" in metadata["fields"]
        assert "v" in metadata["fields"]


class TestNumpyHandler:
    """Tests for NumpyHandler."""

    def test_single_trajectory_file(self, tmp_output):
        """Test numpy handler creates single trajectory.npy."""
        handler = NumpyHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=64,
            figsize=(4, 4),
        )

        data = np.random.rand(16, 16)
        for i in range(5):
            handler.save_frame(data * i, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        handler.finalize()

        assert (tmp_output / "trajectory.npy").exists()
        assert (tmp_output / "times.npy").exists()

    def test_array_shape(self, tmp_output):
        """Test array has shape (T, H, W, F)."""
        handler = NumpyHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=64,
            figsize=(4, 4),
        )

        for i in range(3):
            handler.save_frame(np.ones((16, 16)) * i, "u", i, i * 0.1, 0.0, 1.0, "viridis")
            handler.save_frame(np.ones((16, 16)) * (i + 0.5), "v", i, i * 0.1, 0.0, 1.0, "plasma")

        metadata = handler.finalize()

        trajectory = np.load(tmp_output / "trajectory.npy")
        assert trajectory.shape == (3, 16, 16, 2)  # T, H, W, F
        assert metadata["shape"] == [3, 16, 16, 2]
        assert metadata["fieldOrder"] == ["u", "v"]

    def test_field_order(self, tmp_output):
        """Test field order matches metadata."""
        handler = NumpyHandler()
        handler.initialize(
            tmp_output,
            field_configs=[("X", "viridis"), ("Y", "plasma")],
            dpi=64,
            figsize=(4, 4),
        )

        handler.save_frame(np.ones((8, 8)), "X", 0, 0.0, 0.0, 1.0, "viridis")
        handler.save_frame(np.ones((8, 8)), "Y", 0, 0.0, 0.0, 1.0, "plasma")

        metadata = handler.finalize()

        assert metadata["fieldOrder"] == ["X", "Y"]


class TestMP4Handler:
    """Tests for MP4Handler."""

    def test_initialization(self, tmp_output):
        """Test MP4 handler initializes correctly."""
        handler = MP4Handler(fps=24)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=64,
            figsize=(4, 4),
        )

        assert handler.fps == 24
        assert handler.output_dir == tmp_output
        assert "u" in handler._writers
        assert "v" in handler._writers

    def test_frame_buffering(self, tmp_output):
        """Test frames can be written without buffering in memory."""
        handler = MP4Handler(fps=30)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=32,
            figsize=(2, 2),
        )

        data = np.random.rand(16, 16)
        for i in range(3):
            handler.save_frame(data, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        metadata = handler.finalize()
        assert (tmp_output / "u.mp4").exists()
        assert metadata["format"] == "mp4"

    def test_creates_video_file(self, tmp_output):
        """Test finalize creates MP4 video files."""
        handler = MP4Handler(fps=30)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=32,
            figsize=(2, 2),
        )

        data = np.random.rand(16, 16)
        for i in range(5):
            handler.save_frame(data * (i + 1) / 5, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        metadata = handler.finalize()

        assert (tmp_output / "u.mp4").exists()
        assert metadata["format"] == "mp4"
        assert metadata["fps"] == 30
        assert "u" in metadata["videos"]
        assert metadata["videos"]["u"] == "u.mp4"

    def test_multi_field_videos(self, tmp_output):
        """Test creating videos for multiple fields."""
        handler = MP4Handler(fps=24)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=32,
            figsize=(2, 2),
        )

        for i in range(3):
            data_u = np.random.rand(16, 16)
            data_v = np.random.rand(16, 16)
            handler.save_frame(data_u, "u", i, i * 0.1, 0.0, 1.0, "viridis")
            handler.save_frame(data_v, "v", i, i * 0.1, 0.0, 1.0, "plasma")

        metadata = handler.finalize()

        assert (tmp_output / "u.mp4").exists()
        assert (tmp_output / "v.mp4").exists()
        assert len(metadata["videos"]) == 2


class TestGIFHandler:
    """Tests for GIFHandler."""

    def test_initialization(self, tmp_output):
        """Test GIF handler initializes correctly."""
        handler = GIFHandler(fps=15)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=64,
            figsize=(4, 4),
        )

        assert handler.fps == 15
        assert handler.output_dir == tmp_output
        assert "u" in handler._writers
        assert "v" in handler._writers

    def test_frame_buffering(self, tmp_output):
        """Test frames can be written without buffering in memory."""
        handler = GIFHandler(fps=10)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=32,
            figsize=(2, 2),
        )

        data = np.random.rand(16, 16)
        for i in range(3):
            handler.save_frame(data, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        metadata = handler.finalize()
        assert (tmp_output / "u.gif").exists()
        assert metadata["format"] == "gif"

    def test_creates_gif_file(self, tmp_output):
        """Test finalize creates GIF files."""
        handler = GIFHandler(fps=10)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=32,
            figsize=(2, 2),
        )

        data = np.random.rand(16, 16)
        for i in range(5):
            handler.save_frame(data * (i + 1) / 5, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        metadata = handler.finalize()

        assert (tmp_output / "u.gif").exists()
        assert metadata["format"] == "gif"
        assert metadata["fps"] == 10
        assert "u" in metadata["gifs"]
        assert metadata["gifs"]["u"] == "u.gif"

    def test_gif_is_valid_and_loops(self, tmp_output):
        """Test that GIF is valid and has loop setting."""
        handler = GIFHandler(fps=10)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis")],
            dpi=32,
            figsize=(2, 2),
        )

        data = np.random.rand(16, 16)
        for i in range(3):
            handler.save_frame(data * (i + 1) / 3, "u", i, i * 0.1, 0.0, 1.0, "viridis")

        handler.finalize()

        # Read the GIF back and verify it's valid
        gif_path = tmp_output / "u.gif"
        frames = imageio.mimread(gif_path)
        assert len(frames) == 3
        # Each frame should be RGB
        assert frames[0].ndim == 3
        assert frames[0].shape[2] in (3, 4)  # RGB or RGBA

    def test_multi_field_gifs(self, tmp_output):
        """Test creating GIFs for multiple fields."""
        handler = GIFHandler(fps=10)
        handler.initialize(
            tmp_output,
            field_configs=[("u", "viridis"), ("v", "plasma")],
            dpi=32,
            figsize=(2, 2),
        )

        for i in range(3):
            data_u = np.random.rand(16, 16)
            data_v = np.random.rand(16, 16)
            handler.save_frame(data_u, "u", i, i * 0.1, 0.0, 1.0, "viridis")
            handler.save_frame(data_v, "v", i, i * 0.1, 0.0, 1.0, "plasma")

        metadata = handler.finalize()

        assert (tmp_output / "u.gif").exists()
        assert (tmp_output / "v.gif").exists()
        assert len(metadata["gifs"]) == 2


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_initialization(self, tmp_output):
        """Test OutputManager initialization."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="heat_2024-01-15",
            field_configs=[("u", "viridis")],
        )

        assert manager.folder_name == "heat_2024-01-15"
        assert manager.output_dir == tmp_output / "heat_2024-01-15"
        assert manager.output_formats == ["png"]

    def test_initialization_with_formats(self, tmp_output):
        """Test OutputManager initialization with formats."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test_numpy",
            field_configs=[("u", "viridis")],
            output_formats=["numpy"],
        )

        assert manager.output_formats == ["numpy"]
        assert len(manager.handlers) == 1
        assert isinstance(manager.handlers[0], NumpyHandler)

    def test_save_scalar_frame_png(self, tmp_output, small_grid):
        """Test saving a scalar field frame with PNG format."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-scalar_2024-01-15",
            field_configs=[("u", "turbo")],
            output_formats=["png"],
        )

        field = ScalarField.random_uniform(small_grid)
        field.label = "u"

        manager.compute_range_for_field([field], "u")
        manager.save_frame(field, "u", frame_index=0, simulation_time=0.0)

        # PNG format uses subfolders
        assert (manager.output_dir / "frames" / "u" / "000000.png").exists()

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

        frames = list((manager.output_dir / "frames" / "u").glob("*.png"))
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

        manager.save_all_fields(collection, frame_index=0, simulation_time=0.0)

        assert (manager.output_dir / "frames" / "u" / "000000.png").exists()
        assert (manager.output_dir / "frames" / "v" / "000000.png").exists()

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
        # Check output format metadata
        assert "output" in saved
        assert saved["output"]["formats"] == ["png"]
        assert "png" in saved["output"]

    def test_numpy_format_saves_trajectory(self, tmp_output, small_grid):
        """Test numpy format saves trajectory correctly."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-numpy",
            field_configs=[("u", "viridis"), ("v", "plasma")],
            output_formats=["numpy"],
        )

        states = []
        for i in range(3):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))

        manager.compute_range_for_field(states, "u")
        manager.compute_range_for_field(states, "v")

        for i, state in enumerate(states):
            manager.save_all_fields(state, i, i * 0.1)

        # Finalize should be called via save_metadata
        metadata_path = manager.save_metadata({"test": "data"})

        # Check trajectory was saved
        assert (manager.output_dir / "trajectory.npy").exists()
        assert (manager.output_dir / "times.npy").exists()

        trajectory = np.load(manager.output_dir / "trajectory.npy")
        assert trajectory.shape == (3, 16, 16, 2)

    def test_h5_format_saves_trajectory(self, tmp_output, small_grid):
        """Test h5 format saves trajectory correctly."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-h5",
            field_configs=[("u", "viridis"), ("v", "plasma")],
            output_formats=["h5"],
        )

        states = []
        for i in range(4):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))

        manager.compute_range_for_field(states, "u")
        manager.compute_range_for_field(states, "v")

        for i, state in enumerate(states):
            manager.save_all_fields(state, i, i * 0.1)

        metadata_path = manager.save_metadata({"test": "data"})

        h5_path = manager.output_dir / "trajectory.h5"
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "trajectory" in f
            assert "times" in f
            assert f["trajectory"].shape == (4, 16, 16, 2)
            assert f["times"].shape == (4,)

        with open(metadata_path) as f:
            saved = json.load(f)
        assert saved["output"]["formats"] == ["h5"]
        assert "h5" in saved["output"]

    def test_mp4_format_saves_videos(self, tmp_output, small_grid):
        """Test mp4 format saves video files correctly."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-mp4",
            field_configs=[("u", "viridis"), ("v", "plasma")],
            output_formats=["mp4"],
            fps=24,
        )

        states = []
        for i in range(5):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))

        manager.compute_range_for_field(states, "u")
        manager.compute_range_for_field(states, "v")

        for i, state in enumerate(states):
            manager.save_all_fields(state, i, i * 0.1)

        # Finalize should be called via save_metadata
        metadata_path = manager.save_metadata({"test": "data"})

        # Check videos were saved
        assert (manager.output_dir / "u.mp4").exists()
        assert (manager.output_dir / "v.mp4").exists()

        # Check metadata
        with open(metadata_path) as f:
            saved = json.load(f)
        assert saved["output"]["formats"] == ["mp4"]
        assert "mp4" in saved["output"]
        assert saved["output"]["mp4"]["fps"] == 24
        assert "u" in saved["output"]["mp4"]["videos"]
        assert "v" in saved["output"]["mp4"]["videos"]

    def test_gif_format_saves_gifs(self, tmp_output, small_grid):
        """Test gif format saves GIF files correctly."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-gif",
            field_configs=[("u", "viridis"), ("v", "plasma")],
            output_formats=["gif"],
            fps=15,
        )

        states = []
        for i in range(5):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))

        manager.compute_range_for_field(states, "u")
        manager.compute_range_for_field(states, "v")

        for i, state in enumerate(states):
            manager.save_all_fields(state, i, i * 0.1)

        # Finalize should be called via save_metadata
        metadata_path = manager.save_metadata({"test": "data"})

        # Check GIFs were saved
        assert (manager.output_dir / "u.gif").exists()
        assert (manager.output_dir / "v.gif").exists()

        # Check metadata
        with open(metadata_path) as f:
            saved = json.load(f)
        assert saved["output"]["formats"] == ["gif"]
        assert "gif" in saved["output"]
        assert saved["output"]["gif"]["fps"] == 15
        assert "u" in saved["output"]["gif"]["gifs"]
        assert "v" in saved["output"]["gif"]["gifs"]

    def test_multiple_formats_simultaneously(self, tmp_output, small_grid):
        """Test saving with multiple formats at once (png + numpy)."""
        manager = OutputManager(
            base_path=tmp_output,
            folder_name="test-multi-format",
            field_configs=[("u", "viridis"), ("v", "plasma")],
            output_formats=["png", "numpy"],
        )

        states = []
        for i in range(3):
            u = ScalarField(small_grid, np.ones(small_grid.shape) * i)
            u.label = "u"
            v = ScalarField(small_grid, np.ones(small_grid.shape) * (i + 0.5))
            v.label = "v"
            states.append(FieldCollection([u, v]))

        manager.compute_range_for_field(states, "u")
        manager.compute_range_for_field(states, "v")

        for i, state in enumerate(states):
            manager.save_all_fields(state, i, i * 0.1)

        # Finalize via save_metadata
        metadata_path = manager.save_metadata({"test": "data"})

        # Check PNG files were created
        assert (manager.output_dir / "frames" / "u" / "000000.png").exists()
        assert (manager.output_dir / "frames" / "v" / "000002.png").exists()

        # Check numpy files were created
        assert (manager.output_dir / "trajectory.npy").exists()
        assert (manager.output_dir / "times.npy").exists()

        trajectory = np.load(manager.output_dir / "trajectory.npy")
        assert trajectory.shape == (3, 16, 16, 2)

        # Check metadata contains both formats
        with open(metadata_path) as f:
            saved = json.load(f)
        assert saved["output"]["formats"] == ["png", "numpy"]
        assert "png" in saved["output"]
        assert "numpy" in saved["output"]
        assert saved["output"]["png"]["framesDirectory"] == "frames/"
        assert saved["output"]["numpy"]["trajectoryFile"] == "trajectory.npy"


class TestCreateMetadata:
    """Tests for create_metadata function."""

    def test_create_complete_metadata(self):
        """Test creating complete metadata."""
        preset_meta = PDEMetadata(
            name="test-pde",
            category="test",
            description="Test PDE",
            equations={"u": "laplace(u)"},
            parameters=[PDEParameter("D", "diffusion")],
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
            resolution=[64, 64],  # Now a list for 2D
            bc=BoundaryConfig(
                x_minus="periodic", x_plus="periodic",
                y_minus="periodic", y_plus="periodic"
            ),
            output=OutputConfig(path=Path("./output"), num_frames=100, formats=["png"]),
            seed=42,
            domain_size=[1.0, 1.0],  # Now a list for 2D
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
        assert metadata["simulation"]["ndim"] == 2
        assert len(metadata["frameAnnotations"]) == 2

    def test_create_metadata_with_description(self):
        """Test that description is loaded for real presets."""
        preset_meta = PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott model",
            equations={"u": "D_u*laplace(u)", "v": "D_v*laplace(v)"},
            parameters=[PDEParameter("D_u", "U diffusion")],
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
            resolution=[64, 64],  # Now a list for 2D
            bc=BoundaryConfig(
                x_minus="periodic", x_plus="periodic",
                y_minus="periodic", y_plus="periodic"
            ),
            output=OutputConfig(path=Path("./output"), num_frames=100, formats=["png"]),
            seed=42,
            domain_size=[1.0, 1.0],  # Now a list for 2D
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

        manager.save_all_fields(collection, 0, 0.0)

        # PNG format uses subfolders
        assert (manager.output_dir / "frames" / "u" / "000000.png").exists()
        assert (manager.output_dir / "frames" / "v" / "000000.png").exists()
        assert len(manager.saved_frames) == 1

    def test_filename_format_png(self, tmp_output, small_grid):
        """Test that PNG files follow frames/{field}/{frame:06d}.png format."""
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

        # Check files in subfolders
        for i in range(3):
            assert (manager.output_dir / "frames" / "X" / f"{i:06d}.png").exists()
            assert (manager.output_dir / "frames" / "Y" / f"{i:06d}.png").exists()

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
