import open3d as o3d

PATH = "/home/ferdinand/sam_project/sam_server/converting_worker"#
INPUT_DIR = os.path.join(PATH, "input")
OUTPUT_DIR = os.path.join(PATH, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

in_ply = os.path.join(INPUT_DIR, "gaussian_splat.ply")
out_stl_visual = os.path.join(OUTPUT_DIR, "object_visual.stl")
out_obj_visual = os.path.join(OUTPUT_DIR, "object_visual.obj")
out_stl_collision = os.path.join(OUTPUT_DIR, "object_collision.stl")
out_obj_collision = os.path.join(OUTPUT_DIR, "object_collision.obj")

print(f"Reading point cloud from {in_ply} ...")
pcd = o3d.io.read_point_cloud(in_ply)
assert not pcd.is_empty()
print(f"Loaded point cloud with {len(pcd.points)} points.")

print("Estimating normals ...")
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
print("Mesh cleaned.")

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

print(f"Writing visual mesh to {out_stl_visual} and {out_obj_visual} ...")
o3d.io.write_triangle_mesh(out_stl_visual, visual)
o3d.io.write_triangle_mesh(out_obj_visual, visual)

print(f"Writing collision mesh to {out_stl_collision} and {out_obj_collision} ...")
o3d.io.write_triangle_mesh(out_stl_collision, collision)
o3d.io.write_triangle_mesh(out_obj_collision, collision)

print("OK:", len(mesh.triangles), "triangles in original mesh")



while True:
    if not os.path.exists(os.path.join(INPUT_DIR, "job.txt")):
        time.sleep(0.1)
        continue

    with open(os.path.join(INPUT_DIR, "job.txt")) as f:
        image_path, out_path = f.read().strip().split(",")

    run_sam(image_path, out_path)

    os.remove(os.path.join(INPUT_DIR, "job.txt"))
    open(os.path.join(OUTPUT_DIR, "done.txt"), "w").close()
    