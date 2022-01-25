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
sys.path.append("/home/avena/blenderproc/scripts")

from utils import choose_items_to_load, sample_pose_wrapper, get_objs

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
parser.add_argument('data', nargs='?')
args = parser.parse_args()


def main():
    n = 1
    all_items = get_objs(args.data, None, [])
    consumables_items = get_objs(args.data, "consumables", [])
    containers_items = get_objs(args.data, "containers", [])
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

        table = blenderproc.loader.load_blend("/home/avena/blenderproc/scenes/Bez_fspy.blend")[0]
        table.enable_rigidbody(False, collision_shape='CONVEX_HULL', collision_margin=0.001, mass=5)
        materials = blenderproc.loader.load_ccmaterials("/home/avena/blenderproc/scenes/new_textures2", preload=True)
        table.new_material("Texture")
        for i, material in enumerate(table.get_materials()):
            table.set_material(i, random.choice(materials))
        blenderproc.loader.load_ccmaterials("/home/avena/blenderproc/scenes/new_textures2", fill_used_empty_materials=True)

        hdri = blenderproc.loader.get_random_world_background_hdr_img_path_from_haven("/home/avena/blenderproc/scenes")
        blenderproc.world.set_world_background_hdr_img(hdri)

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
            poi = blenderproc.object.compute_poi(loaded_items)
        if scenario_number == 1:
            pass
        if scenario_number == 2:
            pass


        light_p = blenderproc.types.Light()
        light_p.set_type("SPOT")
        light_p.set_location([0, 0, 5])
        light_p.set_rotation_euler([0, 0, 0])
        light_p.set_energy(200)


        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=1000, image_height=720,
                                                              lens_unit='FOV')

        position = [0, 0, 138]
        rotation = [0, 0, 0]
        # position = [0, 0, 100]
        matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        blenderproc.camera.add_camera_pose(matrix_world)



        # position = [0, 0, 110]
        # matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        # blenderproc.camera.add_camera_pose(matrix_world)

        # blenderproc.material.change_to_texture_less_render(True)
        for item in loaded_items:
            for material in item.get_materials():
                blenderproc.material.add_dust(material, strength=np.random.uniform(0, 0.3))

        position = [30, 30, 90]
        rotation = blenderproc.camera.rotation_from_forward_vec(poi - position)
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