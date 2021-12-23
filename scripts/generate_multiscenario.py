# TODO: Collision shapes: MESHES
# TODO: Plane light
# TODO: 3 scenarios


import blenderproc
import numpy as np
import argparse
import os
import sys
sys.path.append("/home/avena/blenderproc/scripts")


from utils import get_objs, choose_items_to_load, sample_pose_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
parser.add_argument('data', nargs='?')
args = parser.parse_args()


def main():
    n = 10000
    all_items = get_objs(args.data, None, [])
    consumables_items = get_objs(args.data, "consumables", [])
    plane_containers = get_objs(args.data, "containers", ['bowl', 'plate'])
    capable_containers = get_objs(args.data, "containers", ['bowl', 'plate'])

    for _ in range(n):
        # os.system("rm -rf /tmp/blender*")
        scenario_number = np.random.randint(0, 3)

        blenderproc.init()
        blenderproc.utility.reset_keyframes()

        table = blenderproc.loader.load_blend("/home/avena/blenderproc/scenes/Bez_fspy.blend")
        table[0].enable_rigidbody(False, collision_shape='CONVEX_HULL', collision_margin=0.001, mass=5)
        blenderproc.lighting.light_surface(table, -5)

        if scenario_number == 0:
            number_of_items = np.random.randint(10, 25)
            items_to_load, ids_of_items_to_load = choose_items_to_load(all_items, number_of_items)

            sampler = sample_pose_wrapper(table[0], 0, 0)
            loaded_items = [item for item_to_load in items_to_load for item in
                            blenderproc.loader.load_obj(item_to_load)]
            for item, item_id, name in zip(loaded_items, ids_of_items_to_load, items_to_load):
                item.set_cp("category_id", item_id)
                item_name = name.split("/")[-3].split("-")[-1]
                item.set_name(item_name)
                item.enable_rigidbody(True, collision_shape='MESH', collision_margin=0.001)#, mass=500, friction=150)

            loaded_items = blenderproc.object.sample_poses_on_surface(loaded_items, table[0], sampler, min_distance=0.1, max_distance=3)

        if scenario_number == 1:
            n_containers = 5
            container_to_load, id_of_container_to_load = choose_items_to_load(capable_containers, n_containers)
            containers = [item for item_to_load in container_to_load for item in
                          blenderproc.loader.load_obj(item_to_load)]

            for name, container, _id in zip(container_to_load, containers, id_of_container_to_load):
                container.set_cp("category_id", _id)
                container.enable_rigidbody(False, collision_shape='MESH')
                item_name = name.split("/")[-3].split("-")[-1]
                container.set_name(item_name)

            container_sampler = sample_pose_wrapper(table[0], 0, 0)

            containers = blenderproc.object.sample_poses_on_surface(containers, table[0], container_sampler,
                                                                    min_distance=0.1, max_distance=3)
            blenderproc.lighting.light_surface(containers, -1)

            for i in range(len(containers)):
                consumable_to_load, id_of_consumable_to_load = choose_items_to_load(consumables_items, 1)

                if "small" in container_to_load[i]:
                    n_consumables = 2
                else:
                    n_consumables = 4

                consumables = [blenderproc.loader.load_obj(consumable_to_load[0])[0] for c in range(n_consumables)]

                consumables_sampler = sample_pose_wrapper(containers[i], 0, 0)
                consumables = blenderproc.object.sample_poses_on_surface(consumables, containers[i],
                                                                         consumables_sampler, min_distance=0.1,
                                                                         max_distance=1)

                bounding_box = containers[i].get_bound_box()
                points = bounding_box[[0, 1, 4, 5]]
                points = sorted(points, key=lambda x: x[0])
                xy_points = []
                for point in points:
                    xy_points.append(point[:2])

                center = (xy_points[0] + xy_points[3]) / 2

                spawn_locations = [center]

                for k in range(4):
                    spawn_locations.append((xy_points[k] + center) / 2)

                max_z = np.max(bounding_box, axis=0)[2]

                for j in range(n_consumables - len(consumables)):
                    obj = blenderproc.loader.load_obj(consumable_to_load[0])[0]
                    obj.set_location(list(spawn_locations[j % 5]) + [
                        2 * (max_z + np.max(obj.get_bound_box(), axis=0)[2] * (j // 5 + 1))])
                    consumables.append(obj)

                item_name = consumable_to_load[0].split("/")[-3].split("-")[-1]
                for consumable in consumables:
                    consumable.enable_rigidbody(True, collision_shape='CONVEX_HULL')
                    consumable.set_cp("category_id", id_of_consumable_to_load[0])
                    consumable.set_name(item_name)
            blenderproc.object.simulate_physics_and_fix_final_poses(0.5, 2)

        if scenario_number == 2:
            n_containers = 5
            container_to_load, id_of_container_to_load = choose_items_to_load(plane_containers, n_containers)
            containers = [item for item_to_load in container_to_load for item in
                          blenderproc.loader.load_obj(item_to_load)]

            for name, container, _id in zip(container_to_load, containers, id_of_container_to_load):
                container.set_cp("category_id", _id)
                container.enable_rigidbody(False, collision_shape='MESH')
                item_name = name.split("/")[-3].split("-")[-1]
                container.set_name(item_name)

            container_sampler = sample_pose_wrapper(table[0], 0, 0)

            containers = blenderproc.object.sample_poses_on_surface(containers, table[0], container_sampler,
                                                                    min_distance=0.1, max_distance=3)
            blenderproc.lighting.light_surface(containers, -5)

            for i in range(len(containers)):
                consumables_to_load, ids_of_consumable_to_load = choose_items_to_load(consumables_items, 4)

                consumables = [blenderproc.loader.load_obj(consumable_to_load)[0] for consumable_to_load in
                               consumables_to_load]

                for name, consumable, _id in zip(consumables_to_load, consumables, ids_of_consumable_to_load):
                    consumable.set_cp("category_id", _id)
                    consumable.enable_rigidbody(False, collision_shape='CONVEX_HULL')
                    item_name = name.split("/")[-3].split("-")[-1]
                    consumable.set_name(item_name)

                consumables_sampler = sample_pose_wrapper(containers[i], 0, 0)

                blenderproc.object.sample_poses_on_surface(consumables, containers[i], consumables_sampler,
                                                           min_distance=0.1, max_distance=1)
            blenderproc.object.simulate_physics_and_fix_final_poses(0.5, 2)


        light_p = blenderproc.types.Light()
        light_p.set_type("POINT")
        light_p.set_location([0, 0, 0.9])
        light_p.set_rotation_euler([0, 0, 0])
        light_p.set_energy(80)

        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453 / 2, image_width=1000, image_height=720,
                                                              lens_unit='FOV')

        position = [0, 0, 138]
        rotation = [0, 0, np.pi / 2]

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

if __name__ == '__main__':
    main()