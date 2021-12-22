# TODO: Collision shapes: MESHES
# TODO: Plane light
# TODO: 3 scenarios


import blenderproc
import json
import math
import numpy as np
import argparse
import os
import random
import sys
sys.path.append("/home/avena/software/items3/scripts")


from utils import get_all_items_list, choose_items_to_load, sample_pose_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
args = parser.parse_args()



def main():
    n = 10000
    all_items = get_all_items_list("/home/avena/software/items3/loading_dictionaries/items_dictionary.json")
    consumables_items = get_all_items_list("/home/avena/software/items3/loading_dictionaries/consumables_dictionary.json")
    containers_items = get_all_items_list("/home/avena/software/items3/loading_dictionaries/containers_dictionary.json")
    plane_containers = {}
    capable_containers = {}
    for k, v in containers_items.items():
        if "bowl" in k or "plate" in k:
            capable_containers[k] = v
        else:
            plane_containers[k] = v

    for _ in range(n):
        os.system("rm -rf /tmp/blender*")
        scenario_number = np.random.randint(0, 1)

        blenderproc.init()
        blenderproc.utility.reset_keyframes()

        table = blenderproc.loader.load_blend("/home/avena/software/items3/scenes/Bez_fspy.blend")[0]
        table.enable_rigidbody(False, collision_shape='CONVEX_HULL', collision_margin=0.001, mass=5)

        if scenario_number == 0:
            number_of_items = np.random.randint(10, 25)
            items_to_load, ids_of_items_to_load = choose_items_to_load(all_items, number_of_items)

            sampler = sample_pose_wrapper(table, 0, 0)
            loaded_items = [item for item_to_load in items_to_load for item in
                            blenderproc.loader.load_obj(item_to_load)]
            for item, item_id, name in zip(loaded_items, ids_of_items_to_load, items_to_load):
                item.set_cp("category_id", item_id)
                item_name = name.split("/")[-3].split("-")[-1]
                item.set_name(item_name)
                item.enable_rigidbody(True, collision_shape='MESH', collision_margin=0.001)#, mass=500, friction=150)

            loaded_items = blenderproc.object.sample_poses_on_surface(loaded_items, table, sampler, min_distance=0.1, max_distance=3)
            # blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=4,
            #                                                         check_object_interval=1, substeps_per_frame=10)
        if scenario_number == 1:
            pass
        if scenario_number == 2:
            pass


        light_p = blenderproc.types.Light()
        light_p.set_type("AREA")
        light_p.set_location([0, 0, 5])
        light_p.set_rotation_euler([0, 0, 0])
        light_p.set_energy(200)

        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=1000, image_height=720,
                                                              lens_unit='FOV')

        position = [0, 0, 100]
        rotation = [0, 0, 0]

        matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        blenderproc.camera.add_camera_pose(matrix_world)

        blenderproc.renderer.set_noise_threshold(0.1)
        data = blenderproc.renderer.render()
        seg_data = blenderproc.renderer.render_segmap(map_by=["class", "instance"])
        blenderproc.writer.write_coco_annotations(args.output,
                                                  instance_segmaps=seg_data["instance_segmaps"],
                                                  instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                  colors=data["colors"],
                                                  color_file_format="JPEG",
                                                  append_to_existing_output=True)

        # light_p.delete()
if __name__ == '__main__':
    main()