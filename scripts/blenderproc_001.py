import blenderproc
import json
import math
import numpy as np
import argparse
import sys

sys.path.append("/home/avena/software/items3/scripts")

from utils import choose_items_to_load, get_all_items_list, get_objs

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
parser.add_argument('data', nargs='?')
args = parser.parse_args()

global table


def sample_pose(obj_sampled: blenderproc.types.MeshObject) -> None:
    global table
    obj_sampled.set_location(blenderproc.sampler.upper_region(
        objects_to_sample_on=table,
        min_height=0.1,
        max_height=0.2,
        use_ray_trace_check=False
    ))
    obj_sampled.set_rotation_euler(np.random.uniform([0, 0, 0], [0.7, 0.7, np.pi * 2]))


def main():
    global table
    # Load items from file
    items = get_objs(args.data)

    for _ in range(10000):
        # Reset environment
        blenderproc.init()
        blenderproc.utility.reset_keyframes()

        # Sample items to load
        n = 40
        items_to_load, ids_of_loaded_items = choose_items_to_load(items, n)
        # init blenderproc

        # Load the table
        table = blenderproc.loader.load_blend("../scenes/Bez_fspy.blend")[0]
        # Make table static
        table.enable_rigidbody(False)

        # Load sampled items
        loaded_items = [item for item_to_load in items_to_load for item in blenderproc.loader.load_obj(item_to_load)]
        # # Set IDs and enable physics
        for item, item_id, name in zip(loaded_items, ids_of_loaded_items, items_to_load):
            item_name = name.split("/")[-1].split("-")[-1].split(".")[0]
            item.set_name(item_name)
            item.enable_rigidbody(True)
            item.set_cp("category_id", item_id)


        # Place loaded items on the table
        blenderproc.object.sample_poses_on_surface(loaded_items, table, sample_pose, min_distance=0.1, max_distance=10)

        # Simulate physics
        blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=4,
                                                                check_object_interval=0.1, substeps_per_frame=10)

        # Sample light
        light_p = blenderproc.types.Light()
        light_p.set_type("POINT")
        r = np.random.randint(2, 10)
        alpha = np.random.randint(0, 360)
        y = r * math.sin(math.pi / 180 * alpha)
        x = r * math.cos(math.pi / 180 * alpha)
        z = np.random.randint(2, 5)
        light_p.set_location([x, y, z])
        light_p.set_energy(1000)

        # Set camera
        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=2400, image_height=1350,
                                                              lens_unit='FOV')

        # Position the camera
        position = [0, 0, 138]
        rotation = [0, 0, 0]
        matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        blenderproc.camera.add_camera_pose(matrix_world)

        # Render
        # blenderproc.renderer.enable_normals_output()
        blenderproc.renderer.set_samples(30)
        data = blenderproc.renderer.render()
        seg_data = blenderproc.renderer.render_segmap(map_by=["class", "instance"])  # IT'S OK

        blenderproc.writer.write_coco_annotations(args.output,
                                                  instance_segmaps=seg_data["instance_segmaps"],
                                                  instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                  colors=data["colors"],
                                                  append_to_existing_output=True)


if __name__ == "__main__":
    main()

