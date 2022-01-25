import blenderproc
import os
import numpy as np
import json


def get_objs(path: str, category: str = None, items_names_to_exclude: list = []) -> dict:
    item_dictionary = {}
    to_exclude = False
    if category is not None:
        path = os.path.join(path, category)
    for item in os.walk(path):
        for obj in item[2]:
            if obj.endswith(".obj") and obj[0].isalpha():
                to_exclude = False
                for item_to_exclude in items_names_to_exclude:
                    if item_to_exclude in obj:
                        to_exclude = True
                        break
                if not to_exclude:
                    item_dictionary[os.path.join(item[0], obj)] = int(item[0].split("/")[-2].split("-")[0])
    return item_dictionary


def choose_items_to_load(items: dict, n: int):
    ids = list(set(items.values()))
    ids_list = list(np.random.choice(ids, n))
    ids_list = list(map(int, ids_list))

    items_list = []
    for chosen_id in ids_list:
        matching_items = [k for k, v in items.items() if v == chosen_id]
        items_list.append(np.random.choice(matching_items, 1)[0])

    return items_list, ids_list


def get_all_items_list(path_json: str) -> dict:
    with open(path_json, 'r') as f:
        return json.load(f)


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


if __name__ == "__main__":
    d = get_objs("/home/avena/Dropbox/3Dobj")
    print(len(d))