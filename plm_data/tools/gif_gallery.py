"""Build an HTML gallery for PDE GIF outputs."""

from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


_TITLE = "PDE GIF Gallery"


@dataclass(frozen=True)
class GifRun:
    """One PDE output directory and its GIF fields."""

    relative_dir: Path
    label: str
    fields: dict[str, Path]


@dataclass(frozen=True)
class GallerySummary:
    """Summary of one generated gallery."""

    output_path: Path
    num_rows: int
    num_fields: int


def _natural_sort_key(value: str) -> list[str | int]:
    parts = re.split(r"(\d+)", value)
    key: list[str | int] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _run_label(root_dir: Path, directory: Path) -> str:
    relative_dir = directory.relative_to(root_dir)
    if relative_dir != Path("."):
        return relative_dir.as_posix()

    meta_path = directory / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None

        if isinstance(meta, dict):
            category = meta.get("category")
            preset = meta.get("preset")
            if isinstance(category, str) and isinstance(preset, str):
                return f"{category}/{preset}"

    return directory.name


def collect_gif_runs(root_dir: Path) -> list[GifRun]:
    """Return all GIF-bearing directories below ``root_dir``."""

    root_dir = root_dir.resolve()
    grouped: dict[Path, dict[str, Path]] = {}

    for gif_path in root_dir.rglob("*.gif"):
        grouped.setdefault(gif_path.parent, {})[gif_path.stem] = gif_path

    runs = [
        GifRun(
            relative_dir=directory.relative_to(root_dir),
            label=_run_label(root_dir, directory),
            fields=fields,
        )
        for directory, fields in grouped.items()
    ]
    runs.sort(key=lambda run: _natural_sort_key(run.label))
    return runs


def collect_field_names(runs: list[GifRun]) -> list[str]:
    """Return the union of field names across all runs."""

    field_names = {field_name for run in runs for field_name in run.fields}
    return sorted(field_names, key=_natural_sort_key)


def _image_src(gif_path: Path, html_path: Path) -> str:
    rel_path = os.path.relpath(gif_path, start=html_path.parent)
    return quote(Path(rel_path).as_posix(), safe="/")


def _render_html(
    *,
    title: str,
    root_dir: Path,
    output_path: Path,
    runs: list[GifRun],
    field_names: list[str],
) -> str:
    header_cells = "\n".join(
        f'            <th scope="col">{escape(field_name)}</th>'
        for field_name in field_names
    )

    body_rows: list[str] = []
    for run in runs:
        row_cells = []
        for field_name in field_names:
            gif_path = run.fields.get(field_name)
            if gif_path is None:
                row_cells.append('            <td class="empty"></td>')
                continue

            src = _image_src(gif_path, output_path)
            alt = f"{run.label} {field_name}"
            row_cells.append(
                "            <td>\n"
                '              <figure class="gif-card">\n'
                f'                <img src="{src}" alt="{escape(alt)}" loading="lazy">\n'
                "              </figure>\n"
                "            </td>"
            )

        body_rows.append(
            "        <tr>\n"
            f'          <th scope="row"><span class="row-label">{escape(run.label)}</span></th>\n'
            f"{'\n'.join(row_cells)}\n"
            "        </tr>"
        )

    rows_html = "\n".join(body_rows)
    title_text = escape(title)
    root_text = escape(str(root_dir))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title_text}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f4f1e8;
        --panel: rgba(255, 252, 245, 0.9);
        --header: #ded6c4;
        --grid: #cdbfa5;
        --text: #201c15;
        --muted: #6c6457;
        --shadow: 0 18px 40px rgba(61, 46, 24, 0.12);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        background:
          radial-gradient(circle at top left, rgba(207, 189, 149, 0.45), transparent 32%),
          linear-gradient(180deg, #f7f4ec 0%, #efe9db 100%);
        color: var(--text);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      }}

      main {{
        padding: 24px;
      }}

      header {{
        margin-bottom: 18px;
      }}

      h1 {{
        margin: 0 0 6px;
        font-size: clamp(1.8rem, 2.6vw, 2.6rem);
        font-weight: 600;
      }}

      p {{
        margin: 0;
        color: var(--muted);
      }}

      .table-wrap {{
        overflow: auto;
        border: 1px solid rgba(127, 105, 61, 0.18);
        border-radius: 18px;
        background: var(--panel);
        box-shadow: var(--shadow);
      }}

      table {{
        width: max-content;
        min-width: 100%;
        border-collapse: separate;
        border-spacing: 0;
      }}

      th,
      td {{
        border-right: 1px solid var(--grid);
        border-bottom: 1px solid var(--grid);
        padding: 12px;
        vertical-align: top;
        background: rgba(255, 255, 255, 0.72);
      }}

      thead th {{
        position: sticky;
        top: 0;
        z-index: 3;
        background: var(--header);
        text-align: left;
      }}

      tbody th {{
        position: sticky;
        left: 0;
        z-index: 2;
        min-width: 220px;
        background: #f7f0de;
        text-align: left;
      }}

      thead th:first-child {{
        left: 0;
        z-index: 4;
      }}

      tr:last-child td,
      tr:last-child th {{
        border-bottom: 0;
      }}

      th:last-child,
      td:last-child {{
        border-right: 0;
      }}

      .row-label {{
        display: block;
        font-weight: 600;
        word-break: break-word;
      }}

      .gif-card {{
        margin: 0;
      }}

      img {{
        display: block;
        width: min(240px, 28vw);
        min-width: 180px;
        height: auto;
        border-radius: 12px;
        border: 1px solid rgba(83, 66, 38, 0.18);
        background: white;
      }}

      .empty {{
        background: rgba(113, 91, 49, 0.06);
      }}

      @media (max-width: 900px) {{
        main {{
          padding: 14px;
        }}

        th,
        td {{
          padding: 8px;
        }}

        tbody th {{
          min-width: 160px;
        }}

        img {{
          width: min(220px, 52vw);
          min-width: 140px;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>{title_text}</h1>
        <p>{len(runs)} PDE rows, {len(field_names)} fields, scanned from {root_text}</p>
      </header>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">PDE</th>
{header_cells}
            </tr>
          </thead>
          <tbody>
{rows_html}
          </tbody>
        </table>
      </div>
    </main>
  </body>
</html>
"""


def write_gallery_html(
    root_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str = _TITLE,
) -> GallerySummary:
    """Write a GIF gallery HTML page for ``root_dir``."""

    resolved_root = Path(root_dir).resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_root}")

    runs = collect_gif_runs(resolved_root)
    if not runs:
        raise ValueError(f"No GIF files found under {resolved_root}")

    resolved_output = (
        resolved_root / "pde_gif_gallery.html"
        if output_path is None
        else Path(output_path).resolve()
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    field_names = collect_field_names(runs)
    html = _render_html(
        title=title,
        root_dir=resolved_root,
        output_path=resolved_output,
        runs=runs,
        field_names=field_names,
    )
    resolved_output.write_text(html, encoding="utf-8")
    return GallerySummary(
        output_path=resolved_output,
        num_rows=len(runs),
        num_fields=len(field_names),
    )
