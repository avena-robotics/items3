import blenderproc
import json
import numpy as np
import argparse

import sys
sys.path.append("/home/avena/blenderproc/scripts")

from utils import get_objs, choose_items_to_load, sample_pose_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
parser.add_argument('data', nargs="?")
args = parser.parse_args()


def main():
    blenderproc.init()
    blenderproc.utility.reset_keyframes()

    table = blenderproc.loader.load_blend("/home/avena/software/items3/scenes/Bez_fspy.blend")
    table[0].enable_rigidbody(True, collision_shape='CONVEX_HULL')
    blenderproc.lighting.light_surface(table, -5, keep_using_base_color=True)

    consumables_items = get_objs(args.data, "consumables", [])
    containers_items = get_objs(args.data, "containers", ["plate", "bowl"])

    n_containers = 5
    container_to_load, id_of_container_to_load = choose_items_to_load(containers_items, n_containers)
    containers = [item for item_to_load in container_to_load for item in blenderproc.loader.load_obj(item_to_load)]

    for name, container, _id in zip(container_to_load, containers, id_of_container_to_load):
        container.set_cp("category_id", _id)
        container.enable_rigidbody(False, collision_shape='MESH')
        item_name = name.split("/")[-3].split("-")[-1]
        container.set_name(item_name)

    container_sampler = sample_pose_wrapper(table[0], 0, 0)

    blenderproc.object.sample_poses_on_surface(containers, table[0], container_sampler, min_distance=0.1, max_distance=3)
    blenderproc.lighting.light_surface(containers, -5, keep_using_base_color=True)

    for i in range(n_containers):
        consumables_to_load, ids_of_consumable_to_load = choose_items_to_load(consumables_items, 4)

        consumables = [blenderproc.loader.load_obj(consumable_to_load)[0] for consumable_to_load in consumables_to_load]

        for name, consumable, _id in zip(consumables_to_load, consumables, ids_of_consumable_to_load):
            consumable.set_cp("category_id", _id)
            consumable.enable_rigidbody(False, collision_shape='CONVEX_HULL')
            item_name = name.split("/")[-3].split("-")[-1]
            consumable.set_name(item_name)

        consumables_sampler = sample_pose_wrapper(containers[i], 0, 0)

        blenderproc.object.sample_poses_on_surface(consumables, containers[i], consumables_sampler, min_distance=0.1, max_distance=1)

        # for material in consumables[0].get_materials():
        #     material.make_emissive(20)


    blenderproc.object.simulate_physics_and_fix_final_poses(0.5, 2)

    light_p = blenderproc.types.Light()
    light_p.set_type("POINT")
    light_p.set_location([0, 0, 0.9])
    light_p.set_rotation_euler([0, 0, 0])
    light_p.set_energy(80)

    blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=1000, image_height=720,
                                                          lens_unit='FOV')

    position = [0, 0, 80]
    rotation = [0, 0, 0]
    matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
    blenderproc.camera.add_camera_pose(matrix_world)

    blenderproc.renderer.set_samples(30)
    data = blenderproc.renderer.render()
    seg_data = blenderproc.renderer.render_segmap(map_by=["class", "instance", "name"])
    blenderproc.writer.write_coco_annotations(args.output,
                                              instance_segmaps=seg_data["instance_segmaps"],
                                              instance_attribute_maps=seg_data["instance_attribute_maps"],
                                              colors=data["colors"],
                                              color_file_format="JPEG",
                                              append_to_existing_output=True)


if __name__ == '__main__':
    main()