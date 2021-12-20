import blenderproc
import json
import math
import numpy as np
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
args = parser.parse_args()


def main():
    blenderproc.init()
    blenderproc.utility.reset_keyframes()

    table = blenderproc.loader.load_blend("/home/avena/software/items3/scenes/Bez_fspy.blend")
    blenderproc.lighting.light_surface(table, -5, keep_using_base_color=True)
    orange = blenderproc.loader.load_obj("/home/avena/Dropbox/3Dobj/consumables/08-orange/03/orange.obj")[0]
    orange2 = blenderproc.loader.load_obj("/home/avena/Dropbox/3Dobj/consumables/08-orange/03/orange.obj")[0]

    milk = blenderproc.loader.load_obj("/home/avena/Dropbox/3Dobj/consumables/28-milkboxrectangle/01/milk.obj")[0]
    milk_rotation = milk.get_rotation()
    milk_rotation[2] -= 0.4
    milk.set_location([-0.4, -0.3, 0.02])
    milk.set_rotation_euler(milk_rotation)
    orange.set_location([-0.7, 0, 0.03])
    orange2.set_location([-0.52, -0.2, 0.03])
    blenderproc.object.disable_all_rigid_bodies()

    light_p = blenderproc.types.Light()
    light_p.set_type("POINT")
    light_p.set_location([0, 0, 0.9])
    light_p.set_rotation_euler([0, 0, 0])
    light_p.set_energy(100)

    blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=2400, image_height=1350,
                                                          lens_unit='FOV')

    # position = [0, 0, 138]
    rotation = [0, 0, 0]
    position = [0, 0, 100]
    matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
    blenderproc.camera.add_camera_pose(matrix_world)
    blenderproc.renderer.set_samples(100)
    data = blenderproc.renderer.render()
    seg_data = blenderproc.renderer.render_segmap(map_by=["class", "instance", "name"])
    blenderproc.writer.write_coco_annotations(args.output,
                                              instance_segmaps=seg_data["instance_segmaps"],
                                              instance_attribute_maps=seg_data["instance_attribute_maps"],
                                              colors=data["colors"],
                                              color_file_format="JPEG",
                                              append_to_existing_output=True)

    # light_p.delete()


if __name__ == '__main__':
    main()
