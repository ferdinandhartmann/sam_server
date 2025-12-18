# scripts/converter_worker.py

import open3d as o3d
import os
import time

print("\033[92mConversion worker starting...\033[0m")

def run_conversion(input_path, output_dir, done_path):
    out_obj_visual = done_path + "_visual.obj"
    out_obj_collision = done_path + "_collision.obj"

    pcd = o3d.io.read_point_cloud(input_path)
    assert not pcd.is_empty()
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(100)
    print("Normals estimated and oriented.")

    print("Running Poisson surface reconstruction ...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

    print("Cleaning mesh ...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    print("Simplifying mesh for visual model ...")
    visual = mesh.simplify_quadric_decimation(30000)
    print(f"Visual mesh: {len(visual.vertices)} vertices, {len(visual.triangles)} triangles.")

    print("Simplifying mesh for collision model ...")
    collision = mesh.simplify_quadric_decimation(3000)
    print(f"Collision mesh: {len(collision.vertices)} vertices, {len(collision.triangles)} triangles.")

    visual.compute_vertex_normals()
    collision.compute_vertex_normals()
    visual.compute_triangle_normals()
    collision.compute_triangle_normals()

    visual.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

    print(f"Writing visual mesh to {out_obj_visual} ...")
    o3d.io.write_triangle_mesh(out_obj_visual, visual)

    print(f"Writing collision mesh to {out_obj_collision} ...")
    o3d.io.write_triangle_mesh(out_obj_collision, collision)

    print("OK:", len(mesh.triangles), "triangles in original mesh")


PATH = "/home/ferdinand/sam_project/sam_server/worker_data/mesh_worker"
job_name = "job.ply"
done_name = "object"
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
DONE_DIR = os.path.join(PATH, "results")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)
PLY_PATH = os.path.join(INPUT_DIR, job_name)
DONE_PATH = os.path.join(DONE_DIR, done_name)

print("\033[92mConversion worker ready\033[0m")


while True:
    if not os.path.exists(PLY_PATH):
        time.sleep(0.1)
        continue
    
    start_time = time.time()
    print(f"\033[95m[MESH_WORKER] Job started\033[0m")
    run_conversion(PLY_PATH, OUTPUT_DIR, DONE_PATH)
    elapsed_time = time.time() - start_time
    print(f"\033[95m[MESH_WORKER] Time taken: {elapsed_time:.2f} seconds\033[0m")

    os.remove(PLY_PATH)
    