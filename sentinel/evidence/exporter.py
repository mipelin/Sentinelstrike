"""Evidence package zip exporter."""

from __future__ import annotations

import zipfile
from pathlib import Path


def export_zip(run_dir: Path) -> Path | None:
    """Create mission_package.zip from the run directory.

    Returns the zip path, or None if the evidence package directory doesn't exist.
    """
    pkg_dir = run_dir / "mission_package"
    if not pkg_dir.exists():
        return None

    zip_path = run_dir / "mission_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pkg_dir.iterdir()):
            if p.is_file():
                zf.write(p, arcname=f"mission_package/{p.name}")

    return zip_path
