'''
This script generates blueprint-like maps for a Habitat scene.
Examples are included in the same folder.
'''

import torch
from src.env_utils import make_gym_env
import argparse
import numpy as np
from PIL import Image
from habitat.utils.visualizations import maps
from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9

TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [255, 255, 255]
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [50, 255, 0]
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='apartment_1')
    args = parser.parse_args()
    return args


def run(flags):
    flags.device = None
    flags.record = False

    if torch.cuda.is_available():
        print('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        print('Not using CUDA.')
        flags.device = torch.device('cpu')

    flags.env = 'HabitatNav-' + flags.scene
    gym_env = make_gym_env(flags.env)
    print('bounds', gym_env.unwrapped._env.sim.pathfinder.get_bounds())
    gym_env.unwrapped.reset()
    top_down_map = maps.get_topdown_map_from_sim(
            gym_env.unwrapped._env.sim, map_resolution=590
    )
    top_down_map = TOP_DOWN_MAP_COLORS[top_down_map]

    print('resolution', top_down_map.shape)
    im = Image.fromarray(top_down_map)
    im.save(flags.scene + "_top_down_map.png", "png")


if __name__ == '__main__':
    flags = parse_args()
    run(flags)
