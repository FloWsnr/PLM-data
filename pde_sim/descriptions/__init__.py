"""PDE description management utilities."""

from pathlib import Path

# Directory containing description markdown files
DESCRIPTIONS_DIR = Path(__file__).parent


def get_description(preset_name: str) -> str | None:
    """Load the markdown description for a PDE preset.

    Args:
        preset_name: The registered name of the PDE preset (e.g., "heat", "gray-scott").

    Returns:
        The markdown description content, or None if not found.
    """
    md_path = DESCRIPTIONS_DIR / f"{preset_name}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return None


def list_available_descriptions() -> list[str]:
    """List all available PDE descriptions.

    Returns:
        List of preset names that have descriptions.
    """
    return sorted(
        p.stem for p in DESCRIPTIONS_DIR.glob("*.md") if p.stem != "__init__"
    )
