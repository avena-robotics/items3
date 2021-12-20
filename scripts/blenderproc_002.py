import blenderproc
import json
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
args = parser.parse_args()


def main():
    # Load items from file
    container_items = get_all_items_list("/home/avena/PycharmProjects/pythonProject/items/new_items_list_containers.json")
    consumable_items = get_all_items_list("/home/avena/PycharmProjects/pythonProject/items/new_items_list_consumables_and_tools.json")
    for _ in range(100):
        n_containers = 10
        n_consumables = 30
        containers_to_load, ids_of_loaded_containers = choose_items_to_load(container_items, n_containers)
        consumables_to_load, ids_of_loaded_consumables = choose_items_to_load(consumable_items, n_consumables)

        blenderproc.init()
        blenderproc.utility.reset_keyframes()

        table = blenderproc.loader.load_blend("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/Bez_fspy.blend")[0]
        sampler = sample_pose_wrapper(table, 1, 1)
        # Make table static
        table.enable_rigidbody(False)

        loaded_containers = [item for item_to_load in containers_to_load for item in blenderproc.loader.load_obj(item_to_load)]

        for item, item_id, name in zip(loaded_containers, ids_of_loaded_containers, containers_to_load):
            item.set_cp("category_id", item_id)
            item_name = name.split("/")[-3].split("-")[-1]
            item.set_name(item_name)
            item.enable_rigidbody(True, collision_shape='MESH')

        blenderproc.object.sample_poses_on_surface(loaded_containers, table, sampler, min_distance=0.1, max_distance=10)

        # Simulate physics
        blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=2,
                                                                check_object_interval=0.5, substeps_per_frame=3)

        light_p = blenderproc.types.Light()
        light_p.set_type("POINT")
        r = np.random.randint(2, 10)
        alpha = np.random.randint(0, 360)
        y = r * math.sin(math.pi / 180 * alpha)
        x = r * math.cos(math.pi / 180 * alpha)
        z = np.random.randint(2, 5)
        light_p.set_location([x, y, z])
        light_p.set_energy(1000)

        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=2400, image_height=1350,
                                                              lens_unit='FOV')

        position = [0, 0, 138]
        rotation = [0, 0, 0]
        matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        blenderproc.camera.add_camera_pose(matrix_world)

        loaded_consumables = [item for item_to_load in consumables_to_load for item in blenderproc.loader.load_obj(item_to_load)]

        for item, item_id, name in zip(loaded_consumables, ids_of_loaded_consumables, consumables_to_load):
            item.set_cp("category_id", item_id)
            item_name = name.split("/")[-3].split("-")[-1]
            item.set_name(item_name)
            item.enable_rigidbody(True)

        # for loaded_container in loaded_containers:
        #     # if "bowl" in loaded_container.get_name() or "plate" in loaded_container.get_name():
        #     loaded_container.enable_rigidbody(False, collision_shape='MESH')
        #     # else:
        #     #     loaded_container.enable_rigidbody(False)

        containers_dict = {}
        for i in range(n_containers):
            containers_dict[i] = []
        for consumable in loaded_consumables:
            consumable.enable_rigidbody(True)
            on_which_container = np.random.randint(0, n_containers)
            containers_dict[on_which_container].append(consumable)

        for key, val in containers_dict.items():
            sampler = sample_pose_wrapper(loaded_containers[key], np.pi*2, np.pi*2)
            blenderproc.object.sample_poses_on_surface(val, loaded_containers[key], sampler,
                                                       min_distance=0.01, max_distance=10, max_tries=300,
                                                       )

        blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=1.5,
                                                                check_object_interval=0.5, substeps_per_frame=3)

        # blenderproc.renderer.enable_normals_output()
        blenderproc.renderer.set_samples(30)
        data = blenderproc.renderer.render()
        seg_data = blenderproc.renderer.render_segmap(map_by=["class", "instance"])
        blenderproc.writer.write_coco_annotations(args.output,
                                                  instance_segmaps=seg_data["instance_segmaps"],
                                                  instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                  colors=data["colors"],
                                                  color_file_format="JPEG",
                                                  append_to_existing_output=True)


if __name__ == "__main__":
    import time
    now = time.time()
    main()
    print(f"CZAS: {time.time() - now} sekund" )
