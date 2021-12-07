import os
import json


def get_objs(path: str) -> dict:
    item_dictionary = {}
    for item in os.walk(path):
        for obj in item[2]:
            if obj.endswith(".obj") and obj[0].isdigit():
                item_dictionary[os.path.join(item[0], obj)] = int(obj.split("-")[0])
    return item_dictionary



if __name__ == "__main__":
    # items = get_objs("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/obj/")
    #
    # with open("items/containers_json.json", "w") as f:
    #     json.dump(items, f, indent=2)
    items_dict = {}
    for parent in os.walk("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/obj/containers"):
        for child in parent[2]:
            if child.endswith(".obj") and child[0].isalpha():
                items_dict[os.path.join(parent[0], child)] = int(parent[0].split("/")[-2].split("-")[0])

    # for parent in os.walk("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/obj/tools"):
    #     for child in parent[2]:
    #         if child.endswith(".obj") and child[0].isalpha():
    #             items_dict[os.path.join(parent[0], child)] = int(parent[0].split("/")[-2].split("-")[0])

    with open("../items/new_items_list_containers.json", "w") as f:
        json.dump(items_dict, f, indent=2)
