"""Mesh conversion worker.

This worker loads Open3D once, then watches the shared jobs directory for
reconstruction results. When a ``*.ply`` file becomes available it will produce
OBJ and STL meshes for both the visual and collision representations.
"""

from __future__ import annotations

import time
from pathlib import Path

import open3d as o3d

from config import CONVERSION_STAGE, WORKER_LOG_PREFIX, WORKER_POLL_SECONDS
from job_state import claim_next_job, prerequisites_for, set_artifacts, set_stage_status


def convert_mesh(job_dir: Path) -> None:
    ply_path = job_dir / "object.ply"
    output_dir = job_dir / "meshes"
    output_dir.mkdir(exist_ok=True)

    visual_stl = output_dir / "object_visual.stl"
    visual_obj = output_dir / "object_visual.obj"
    collision_stl = output_dir / "object_collision.stl"
    collision_obj = output_dir / "object_collision.obj"

    print(f"{WORKER_LOG_PREFIX} [converter] reading point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty")

    print(f"{WORKER_LOG_PREFIX} [converter] estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    print(f"{WORKER_LOG_PREFIX} [converter] running Poisson reconstruction")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    print(f"{WORKER_LOG_PREFIX} [converter] cleaning mesh")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    print(f"{WORKER_LOG_PREFIX} [converter] simplifying visual mesh")
    visual = mesh.simplify_quadric_decimation(30000)
    print(f"{WORKER_LOG_PREFIX} [converter] simplifying collision mesh")
    collision = mesh.simplify_quadric_decimation(3000)

    for m in (visual, collision):
        m.compute_vertex_normals()
        m.compute_triangle_normals()

    visual.paint_uniform_color([0.7, 0.7, 0.7])

    print(f"{WORKER_LOG_PREFIX} [converter] writing {visual_stl} and {visual_obj}")
    o3d.io.write_triangle_mesh(str(visual_stl), visual)
    o3d.io.write_triangle_mesh(str(visual_obj), visual)

    print(f"{WORKER_LOG_PREFIX} [converter] writing {collision_stl} and {collision_obj}")
    o3d.io.write_triangle_mesh(str(collision_stl), collision)
    o3d.io.write_triangle_mesh(str(collision_obj), collision)

    set_artifacts(
        job_dir,
        visual_stl=str(visual_stl),
        visual_obj=str(visual_obj),
        collision_stl=str(collision_stl),
        collision_obj=str(collision_obj),
    )


def main() -> None:
    while True:
        job_dir = claim_next_job(CONVERSION_STAGE, prerequisites_for(CONVERSION_STAGE))
        if not job_dir:
            time.sleep(WORKER_POLL_SECONDS)
            continue

        try:
            convert_mesh(job_dir)
        except Exception as exc:  # noqa: BLE001 - surface worker error for visibility
            print(f"{WORKER_LOG_PREFIX} [converter] failed: {exc}")
            set_stage_status(job_dir, CONVERSION_STAGE, "failed", error=str(exc))
        else:
            set_stage_status(job_dir, CONVERSION_STAGE, "done")


if __name__ == "__main__":
    main()
    
