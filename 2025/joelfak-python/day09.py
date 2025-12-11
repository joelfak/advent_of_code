#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest
import numpy as np
from itertools import zip_longest

@hf.timing
def part1(data):
    input = np.array([[int(v) for v in r.split(',')] for r in data])
    areas = np.prod(np.array([np.abs(np.subtract.outer(input[:,n],input[:,n])+1) for n in range(2)]),0)
    return np.max(areas)

@hf.timing
def part2(data):
    input = np.array([[int(v) for v in r.split(',')] for r in data])
    areas = np.prod(np.array([np.abs(np.subtract.outer(input[:,n],input[:,n]))+1 for n in range(2)]),0)

    max_x, max_y = np.max(input, 0)
    floor_map = np.zeros((max_y+2, max_x+2), dtype=int)
    floor_map[input[:,[1]], input[:,[0]]] = 1

    red_tiles_indices = input[:,[1,0]]
    red_tiles_indices_shifted = red_tiles_indices.take(range(1, len(red_tiles_indices)+1),
                                                       axis=0,
                                                       mode="wrap")

    # Fill borders
    border_coords = []
    for start_tile, end_tile in zip(red_tiles_indices, red_tiles_indices_shifted):
        range_row = range(min(start_tile[0], end_tile[0]), max(start_tile[0], end_tile[0]+1))
        range_col = range(min(start_tile[1], end_tile[1]), max(start_tile[1], end_tile[1]+1))
        static_coord = range_row.start if len(range_row) == 1 else range_col.start 
        border_coords.extend(zip_longest(range_row, range_col, fillvalue=static_coord))
    border_rows, border_cols = list(zip(*border_coords))

    # Fill outside
    floor_map[border_rows, border_cols] = 2
    num_rows, num_cols = floor_map.shape
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if (floor_map[row_idx, col_idx] == 0) and \
                ((row_idx == 0 or floor_map[row_idx-1, col_idx] == -1) or
                 (col_idx == 0 or floor_map[row_idx, col_idx-1] == -1)):
                floor_map[row_idx, col_idx] = -1

    # Get areas of tiled rectangles
    red_green_tiles = np.where(floor_map >= 0,1,0)
    red_green_tile_areas = np.zeros(areas.shape, dtype=int)
    for tile_idx_1, (col1, row1) in enumerate(input[:,:]):
        for tile_idx_2, (col2, row2) in enumerate(input[:,:]):
            col_start, col_end = sorted([col1, col2])
            row_start, row_end = sorted([row1, row2])
            num_cols = col_end - col_start + 1
            num_rows = row_end - row_start + 1
            red_green_tile_areas[tile_idx_1, tile_idx_2] = np.sum(red_green_tiles[row_start:row_end+1, col_start:col_end+1])

    return max(areas[areas==red_green_tile_areas])

## Unit tests ########################################################

@pytest.fixture
def input():
    return ["7,1",
            "11,1",
            "11,7",
            "9,7",
            "9,5",
            "2,5",
            "2,3",
            "7,3"]

@pytest.mark.parametrize("test_input,expected", [(35, 35),
                                                 (26, 26)])
def test_help_function(test_input, expected):
    assert test_input == expected

def test_part1(input):
    assert part1(input) == 50

def test_part2(input):
    assert part2(input) == 24

## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    print(f"Part1 result: {part1(input.copy())}")
    print(f"Part2 result: {part2(input.copy())}")
