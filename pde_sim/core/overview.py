"""Generate HTML overview of GIF simulations."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationInfo:
    """Info about a discovered simulation."""

    folder: Path
    preset: str
    name: str
    gif_files: dict[str, Path]  # field_name -> gif_path
    spacetime_files: dict[str, Path]  # field_name -> spacetime_png_path
    metadata: dict


def discover_simulations(output_dir: Path) -> list[SimulationInfo]:
    """Discover all simulation folders containing GIF files.

    Args:
        output_dir: Root output directory to scan

    Returns:
        List of SimulationInfo for each discovered simulation
    """
    simulations = []

    # Find all metadata.json files
    for metadata_path in output_dir.rglob("metadata.json"):
        folder = metadata_path.parent

        # Check for GIF files and spacetime PNGs
        gif_files = list(folder.glob("*.gif"))
        spacetime_files = list(folder.glob("*_spacetime.png"))

        if not gif_files and not spacetime_files:
            continue

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Build gif_files dict mapping field name to path
        gif_dict = {}
        for gif_path in gif_files:
            field_name = gif_path.stem  # e.g., "u.gif" -> "u"
            gif_dict[field_name] = gif_path

        # Build spacetime_files dict mapping field name to path
        spacetime_dict = {}
        for st_path in spacetime_files:
            # e.g., "u_spacetime.png" -> "u"
            field_name = st_path.stem.removesuffix("_spacetime")
            spacetime_dict[field_name] = st_path

        # Extract preset name and folder name
        preset = metadata.get("preset", "unknown")
        folder_name = metadata.get("folder_name", folder.name)

        simulations.append(
            SimulationInfo(
                folder=folder,
                preset=preset,
                name=folder_name,
                gif_files=gif_dict,
                spacetime_files=spacetime_dict,
                metadata=metadata,
            )
        )

    # Sort by preset name, then folder name
    simulations.sort(key=lambda s: (s.preset, s.name))

    return simulations


def generate_html(
    simulations: list[SimulationInfo],
    output_path: Path,
    title: str = "Simulation Overview",
) -> None:
    """Generate HTML overview document.

    Args:
        simulations: List of discovered simulations
        output_path: Path to write HTML file
        title: Title for the HTML document
    """
    # Make GIF paths relative to the HTML file location
    html_dir = output_path.parent.resolve()

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"  <title>{title}</title>",
        "  <style>",
        _get_css(),
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        f"  <p class='summary'>Found {len(simulations)} simulations</p>",
    ]

    # Group simulations by preset
    by_preset: dict[str, list[SimulationInfo]] = {}
    for sim in simulations:
        by_preset.setdefault(sim.preset, []).append(sim)

    for preset, sims in sorted(by_preset.items()):
        html_parts.append(f"  <div class='preset-group'>")
        html_parts.append(f"    <h2 class='preset-title'>{preset}</h2>")

        for sim in sims:
            html_parts.append(_render_simulation(sim, html_dir))

        html_parts.append("  </div>")

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    output_path.write_text("\n".join(html_parts))


def _get_css() -> str:
    """Return CSS styles for the HTML document."""
    return """
    * {
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      max-width: 1800px;
      margin: 0 auto;
      padding: 20px;
      background: #1a1a2e;
      color: #eee;
    }
    h1 {
      border-bottom: 2px solid #4a4a6a;
      padding-bottom: 10px;
      color: #fff;
    }
    .summary {
      color: #888;
      margin-bottom: 30px;
    }
    .preset-group {
      margin-bottom: 40px;
    }
    .preset-title {
      background: #2a2a4a;
      padding: 10px 15px;
      border-radius: 6px;
      margin-bottom: 15px;
      color: #8be9fd;
    }
    .simulation-row {
      background: #252540;
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 15px;
    }
    .simulation-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid #3a3a5a;
    }
    .simulation-name {
      font-weight: 600;
      color: #f8f8f2;
      font-size: 1.1em;
    }
    .simulation-meta {
      font-size: 0.85em;
      color: #888;
    }
    .fields-container {
      display: flex;
      flex-wrap: wrap;
      gap: 15px;
    }
    .field-cell {
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #1a1a2e;
      border-radius: 6px;
      padding: 10px;
    }
    .field-cell img {
      max-width: 300px;
      height: auto;
      border-radius: 4px;
      border: 1px solid #3a3a5a;
    }
    .field-cell img.spacetime {
      max-width: 500px;
    }
    .field-label {
      margin-top: 8px;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 0.9em;
      color: #50fa7b;
    }
    """


def _render_simulation(sim: SimulationInfo, html_dir: Path) -> str:
    """Render a single simulation row."""
    lines = []
    lines.append("    <div class='simulation-row'>")

    # Header with name and metadata
    resolution = sim.metadata.get("simulation", {}).get("resolution", [])
    t_end_raw = sim.metadata.get("simulation", {}).get("totalTime", "?")
    t_end = round(t_end_raw, 4) if isinstance(t_end_raw, float) else t_end_raw
    num_frames = sim.metadata.get("simulation", {}).get("numFrames", "?")

    res_str = "x".join(str(r) for r in resolution) if resolution else "?"

    lines.append("      <div class='simulation-header'>")
    lines.append(f"        <span class='simulation-name'>{sim.name}</span>")
    lines.append(
        f"        <span class='simulation-meta'>{res_str} | t={t_end} | {num_frames} frames</span>"
    )
    lines.append("      </div>")

    # Fields container
    ndim = sim.metadata.get("simulation", {}).get("ndim", 2)
    is_1d = ndim == 1

    lines.append("      <div class='fields-container'>")

    # For 1D simulations, prefer spacetime diagrams over GIFs
    if is_1d and sim.spacetime_files:
        for field_name in sorted(sim.spacetime_files.keys()):
            st_path = sim.spacetime_files[field_name]

            try:
                rel_path = st_path.resolve().relative_to(html_dir)
            except ValueError:
                rel_path = st_path.resolve()

            lines.append("        <div class='field-cell'>")
            lines.append(
                f"          <img class='spacetime' src='{rel_path}' alt='{field_name} spacetime' loading='lazy'>"
            )
            lines.append(f"          <span class='field-label'>{field_name} (spacetime)</span>")
            lines.append("        </div>")
    else:
        # Sort fields to have consistent ordering
        for field_name in sorted(sim.gif_files.keys()):
            gif_path = sim.gif_files[field_name]

            # Make path relative to HTML file
            try:
                rel_path = gif_path.resolve().relative_to(html_dir)
            except ValueError:
                # If can't make relative, use absolute path
                rel_path = gif_path.resolve()

            lines.append("        <div class='field-cell'>")
            lines.append(
                f"          <img src='{rel_path}' alt='{field_name}' loading='lazy'>"
            )
            lines.append(f"          <span class='field-label'>{field_name}</span>")
            lines.append("        </div>")

    lines.append("      </div>")
    lines.append("    </div>")

    return "\n".join(lines)


def _copy_assets(
    simulations: list[SimulationInfo], overview_dir: Path
) -> list[SimulationInfo]:
    """Copy GIF and spacetime PNG files into the overview directory.

    Assets are organized as overview_dir/preset/sim_name/.
    """
    updated = []
    for sim in simulations:
        sim_dir = overview_dir / sim.preset / sim.name
        sim_dir.mkdir(parents=True, exist_ok=True)

        new_gif_files = {}
        for field_name, gif_path in sim.gif_files.items():
            dest = sim_dir / gif_path.name
            shutil.copy2(gif_path, dest)
            new_gif_files[field_name] = dest

        new_spacetime_files = {}
        for field_name, st_path in sim.spacetime_files.items():
            dest = sim_dir / st_path.name
            shutil.copy2(st_path, dest)
            new_spacetime_files[field_name] = dest

        updated.append(
            SimulationInfo(
                folder=sim.folder,
                preset=sim.preset,
                name=sim.name,
                gif_files=new_gif_files,
                spacetime_files=new_spacetime_files,
                metadata=sim.metadata,
            )
        )

    return updated


def generate_overview(output_dir: Path, html_path: Path, title: str) -> int:
    """Main entry point: discover simulations, copy GIFs, and generate HTML.

    Creates a self-contained overview/ folder inside output_dir with copied GIFs
    and the HTML file, so it can be easily moved around.

    Args:
        output_dir: Directory to scan for simulations
        html_path: Path to write HTML file
        title: Title for the HTML document

    Returns:
        Number of simulations found
    """
    simulations = discover_simulations(output_dir)

    if not simulations:
        return 0

    # Create overview directory next to the HTML file
    overview_dir = html_path.parent
    overview_dir.mkdir(parents=True, exist_ok=True)

    # Copy GIFs and spacetime PNGs into the overview directory
    simulations = _copy_assets(simulations, overview_dir)

    generate_html(simulations, html_path, title)
    return len(simulations)
