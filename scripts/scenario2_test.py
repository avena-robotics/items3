import blenderproc
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
args = parser.parse_args()


def sample_pose_wrapper(obj_parent: blenderproc.types.MeshObject, d1_max, d2_max):
    def sample_pose_inside(obj_sampled_inside):
        obj_sampled_inside.set_location(blenderproc.sampler.upper_region(
            objects_to_sample_on=obj_parent,
            min_height=0.2,
            max_height=0.4,
            use_ray_trace_check=True,
            upper_dir=[0.0, 0.0, 1.0],
            use_upper_dir=True
        ))
        obj_sampled_inside.set_rotation_euler(np.random.uniform([np.pi/2, 0, 0], [np.pi/2, d2_max, np.pi * 2]))
    return sample_pose_inside

def get_all_items_list(path_json: str) -> dict:
    with open(path_json, 'r') as f:
        return json.load(f)


def choose_items_to_load(items: dict, n: int) -> (list, list):
    ids = list(set(items.values()))
    ids_list = list(np.random.choice(ids, n))
    ids_list = list(map(int, ids_list))

    items_list = []
    for chosen_id in ids_list:
        matching_items = [k for k, v in items.items() if v == chosen_id]
        items_list.append(np.random.choice(matching_items, 1)[0])

    return items_list, ids_list


def main():
    blenderproc.init()
    blenderproc.utility.reset_keyframes()

    table = blenderproc.loader.load_blend("/home/avena/software/items3/scenes/Bez_fspy.blend")
    table[0].enable_rigidbody(True, collision_shape='CONVEX_HULL')
    blenderproc.lighting.light_surface(table, -5, keep_using_base_color=True)

    consumables_items = get_all_items_list("/home/avena/software/items3/loading_dictionaries/consumables_all_dictionary.json")
    containers_items = get_all_items_list("/home/avena/software/items3/loading_dictionaries/containers_flat_dictionary.json")

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

        # if "small" in container_to_load[i]:
        #     n_consumables = 2
        # else:
        #     n_consumables = 4

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

        # bounding_box = containers[i].get_bound_box()
        # points = bounding_box[[0, 1, 4, 5]]
        # points = sorted(points, key=lambda x: x[0])
        # xy_points = []
        # for point in points:
        #     xy_points.append(point[:2])
        #
        # center = (xy_points[0] + xy_points[3]) / 2
        #
        # spawn_locations = [center]
        #
        # for k in range(4):
        #     spawn_locations.append((xy_points[k] + center) / 2)
        #
        # max_z = np.max(bounding_box, axis=0)[2]
        #
        # for j in range(n_consumables - len(consumables)):
        #     obj = blenderproc.loader.load_obj(consumable_to_load[0])[0]
        #     obj.set_location(list(spawn_locations[j % 5]) + [2 * (max_z + np.max(obj.get_bound_box(), axis=0)[2] * (j // 5 + 1))])
        #     consumables.append(obj)
        #
        # item_name = consumable_to_load[0].split("/")[-3].split("-")[-1]
        # for consumable in consumables:
        #     consumable.enable_rigidbody(True, collision_shape='CONVEX_HULL')
        #     consumable.set_cp("category_id", id_of_consumable_to_load[0])
        #     consumable.set_name(item_name)

    blenderproc.object.simulate_physics_and_fix_final_poses(0.5, 2)

    light_p = blenderproc.types.Light()
    light_p.set_type("POINT")
    light_p.set_location([0, 0, 0.9])
    light_p.set_rotation_euler([0, 0, 0])
    light_p.set_energy(80)

    blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=1000, image_height=720,
                                                          lens_unit='FOV')

    position = [0, 0, 70]
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