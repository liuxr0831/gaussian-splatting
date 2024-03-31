# Based on https://github.com/shumash/gaussian-splatting/blob/mshugrina/interactive/interactive.ipynb
print("Importing required packages...")

from argparse import ArgumentParser
import json
import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm

from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration


#torch.cuda.set_per_process_memory_fraction(0.9, 0)

#######################################################################

# parse arguments

parser = ArgumentParser("")
parser.add_argument("--model_path", "-m", required=True, type=str)
parser.add_argument("--frame_json_file_path", "-f", required=True, type=str)
parser.add_argument("--background_color", "-bg", default='0,0,0', type=str)
args = parser.parse_args()




#######################################################################

# Initialize required parameters for rendering

# Placeholder for render
class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

pipeline = PipelineParams()


# background color for rendering
colors = args.background_color.split(',')
if not len(colors)==3:
    print("Background color not specified as 3 comma-separated numbers between 0 and 1.")
    exit()
try:
    bg_color = np.array([float(colors[0]), float(colors[1]), float(colors[2])])
    if any(bg_color < 0) or any(bg_color > 1):
        raise Exception()
except:
    print("Background color not specified as 3 comma-separated numbers between 0 and 1.")
    exit()
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")




#######################################################################

# Load the GS model

print("Loading gaussian splatting model...")

# Find max iteration model (which should have the best quality)
model_path = args.model_path
iteration = -1
if not os.path.exists(model_path):
    print("The specified model_path does not exists")
    exit()

try:
    point_cloud_path = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(point_cloud_path)
    point_cloud_path = os.path.join(point_cloud_path, f"iteration_{iteration}", "point_cloud.ply")

    # Load guassians
    GS_model = GaussianModel(3)
    GS_model.load_ply(point_cloud_path)
except Exception as e:
    print("An error occurred while loading the gaussian splatting model.")
    print(e)
    exit()



#######################################################################

# Read in the json files that define the camera params for each frame
# The json file should be a list of dictionaries, where each dictionary
# should have the fields 'width' (int), 'height' (int), 'fx' (float), 
# 'fy' (float), (these four defines the camera intrinsic), 'position' 
# (a list containing 3 float, i.e. the translation vector), 'rotation' 
# (a 3-by-3 list containing 9 floats, i.e. rotation matrix) (these two 
# defines the camera extrinsic). 
# After reading the file, we render the frames.

frame_json_files = [args.frame_json_file_path]

for frame_json_file in frame_json_files:

    # load file and read the camera configuration for each frame
    with open(frame_json_file, 'rb') as f:
        all_frames_cam = json.load(f)


    # create frame image dir
    frame_file_name_prefix = '.'.join(os.path.basename(frame_json_file).split('.')[:-1])
    frame_json_file_dir = os.path.dirname(frame_json_file)
    frame_file_dir = os.path.join(frame_json_file_dir, frame_file_name_prefix)
    os.makedirs(frame_file_dir, exist_ok = True)


    # render each frame
    print("Rendering frames in {}...".format(frame_json_file))
    for frame_i in tqdm(range(len(all_frames_cam))):
        # read extrinsic
        rot_mat = all_frames_cam[frame_i].get('rotation')
        pos_vec = all_frames_cam[frame_i].get('position')
        if rot_mat is None or pos_vec is None:
            print("Frame {} cannot be rendered because it does not have rotation \
                   matrix or position vector defined".format(frame_i))
            continue

        # compute inverse to fit with GS pipeline
        tmp = np.zeros((4,4))
        tmp[3,3] = 1
        tmp[:3, :3] = rot_mat
        tmp[:3,3] = pos_vec
        cam_to_world = np.linalg.inv(tmp)
        #cam_to_world = tmp

        # extract rotation and translation part
        rot_GS = cam_to_world[:3, :3].transpose()
        pos_GS = cam_to_world[:3, 3]

        # extract intrinsic part
        width_GS = all_frames_cam[frame_i].get('width')
        height_GS = all_frames_cam[frame_i].get('height')
        fx = all_frames_cam[frame_i].get('fx')
        fy = all_frames_cam[frame_i].get('fy')
        if width_GS is None or height_GS is None or fx is None or fy is None:
            print("Frame {} cannot be rendered because it does not have width, \
                   height, focal length x, or focal length y defined".format(frame_i))
            continue
        FoV_x_GS = focal2fov(fx, width_GS)
        FoV_y_GS = focal2fov(fy, height_GS)

        # create GS camera object
        cam_GS = Camera(colmap_id=0, R=rot_GS, T=pos_GS, FoVx=FoV_x_GS, FoVy=FoV_y_GS, \
                        image=torch.zeros((3, height_GS, width_GS)), gt_alpha_mask=None, \
                        image_name='{}_{:08d}'.format(frame_file_name_prefix, frame_i), \
                        uid=0)

        # render and save
        rendering = render(cam_GS, GS_model, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(frame_file_dir, all_frames_cam[frame_i].get('name')))
