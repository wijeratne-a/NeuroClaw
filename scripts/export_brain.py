#!/usr/bin/env python3
"""
Export fsaverage5 pial surface (both hemispheres) to GLB for the React frontend.

Requires: nilearn, trimesh (see pyproject.toml).

Usage from repo root:
  python scripts/export_brain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "frontend" / "public" / "brain.glb"


def main() -> None:
    try:
        import trimesh
        from nilearn import datasets
        from nilearn.surface import load_surf_mesh
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Install with: pip install trimesh nilearn")

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    coords_l, faces_l = load_surf_mesh(fsaverage.pial_left)
    coords_r, faces_r = load_surf_mesh(fsaverage.pial_right)

    mesh_l = trimesh.Trimesh(vertices=coords_l, faces=faces_l)
    mesh_r = trimesh.Trimesh(vertices=coords_r, faces=faces_r)
    brain = trimesh.util.concatenate([mesh_l, mesh_r])

    brain.vertices -= brain.center_mass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    brain.export(str(OUT))
    print(f"Exported to {OUT}")


if __name__ == "__main__":
    main()
