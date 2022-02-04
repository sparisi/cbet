'''
This script generates heatmaps used in the paper to show Habitat state visitation counts.
It needs the top-down map of the scene and some map info (generated/printed by make_topdown_map.py)
and the pickles with the visitation counts (see
https://github.com/sparisi/cbet/blob/67fe0b2544fc2c5bd8d3ae93cd5bcc980fc515b2/src/env_utils.py#L423).
By default, pickles are named 0.pickle, 1.pickle, ... up to n.pickle, where n
is the number of actors that collected the data during training.
'''

from collections import Counter
import pickle5 as pickle
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

from PIL import Image
from typing import Tuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='apartment_0')
    parser.add_argument('--grid_cell_size', type=int, default=6)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--out', type=str, default='test')
    args = parser.parse_args()
    return args

def rate_to_value(rate): # Tune to change the color scale
    # return -255 * (np.log(rate) / 4 + 1) + 50
    return -255 * np.log(rate)

def make_habitat_heatmap(counts, top_down_map, name):
    sum_count = np.sum(np.fromiter(counts.values(), dtype=float))
    rates = {}
    for position in counts.keys():
        visit_rate = counts[position] * 1.0 / sum_count
        rates[position] = visit_rate

    rate0=max(rates.values())
    rate1=min(rates.values())
    print('Min visit rate:', rate0, 'pixel value:', rate_to_value(rate0))
    print('Max visit rate:', rate1, 'pixel value:', rate_to_value(rate1))

    for position, rate in rates.items():
        x, y = position
        value = rate_to_value(rate)
        w = int(args.grid_cell_size / 2 - 1)
        top_down_map[x-w:x+w,y-w:y+w, 1] = value
        top_down_map[x-w:x+w,y-w:y+w, 2] = value

# Borrowed from habitat
def to_grid(
    realworld_x: float,
    realworld_y: float,
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y

args = parse_args()

# Scene-specific coordinates and resolution (printed by make_topdown_map.py)
if args.scene == 'apartment_0':
    lower_bound = [-3.2927506, -1.5747652, -4.9503818]
    upper_bound = [6.07346, 5.3709807, 9.816713]
    grid_resolution = (930, 590)
elif args.scene == 'apartment_1':
    lower_bound = [-2.5994487, -1.8001366, -6.9427214]
    upper_bound = [8.141758, 2.743783, 0.9603044]
    grid_resolution = (590, 801)
elif args.scene == 'apartment_2':
    lower_bound = [-2.6598349, -1.6610725, -8.48371]
    upper_bound = [6.777131, 2.746698, 1.7142575]
    grid_resolution = (637, 590)
elif args.scene == 'room_0':
    lower_bound = [-0.8793944, -1.527363, -3.5122514]
    upper_bound = [6.8852057, 2.8804069, 1.1859794]
    grid_resolution = (590, 975)
elif args.scene == 'room_1':
    lower_bound = [-5.402722 , -1.4079537, -2.689094]
    upper_bound = [1.2436305, 2.9451954, 3.038546]
    grid_resolution = (590, 684)
elif args.scene == 'room_2':
    lower_bound = [-0.81706226, -2.908131, -1.6999526]
    upper_bound = [5.953264, 2.2861283, 3.2454479]
    grid_resolution = (590, 807)
elif args.scene == 'hotel_0':
    lower_bound = [-2.9058013, -1.0352085, -2.5156698]
    upper_bound = [5.6849318, 3.3101344, 1.6605881]
    grid_resolution = (590, 1213)
elif args.scene == 'frl_apartment_0':
    lower_bound = [-1.0419858, -1.6410627, -3.464557]
    upper_bound = [6.204813 , 3.0397148, 9.190308]
    grid_resolution = (1030, 590)
elif args.scene == 'office_3':
    lower_bound = [-5.111591 , -1.22069  , -3.2651799]
    upper_bound = [3.532915 , 3.4816294, 5.9394565]
    grid_resolution = (628, 590)
else:
    raise NotImplementedError("This scene has not been implemented.")

visits = dict()
for i in range(args.n):
    try:
        with open(str(i) + '.pickle', 'rb') as handle:
            v = pickle.load(handle)
            visits = dict(Counter(visits) + Counter(v))
    except:
        print("Couldn't load pickle", i)

counts_by_pixel = {}
state_pixels = []

for position, count in visits.items():
    x, y = to_grid(position[1], position[0])
    x = args.grid_cell_size * round(x / args.grid_cell_size)
    y = args.grid_cell_size * round(y / args.grid_cell_size)
    counts_by_pixel[(x,y)] = count

top_down_map = Image.open(os.path.join(args.scene + '_top_down_map.png'), mode='r')
top_down_map = np.asarray(top_down_map).copy()

top_down_map = np.rot90(top_down_map, k=1)
make_habitat_heatmap(counts_by_pixel, top_down_map, 'test_cbet')

plt.imshow(top_down_map)
plt.show()

im = Image.fromarray(top_down_map)
im.save(args.out + '.png', "png")
