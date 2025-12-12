#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest
import numpy as np
from itertools import zip_longest
from collections import deque

UNKNOWN = 0
RED_TILE = 1
BORDER = 2
OUTSIDE = -1

@hf.timing
def part1(data):
    input = np.array([[int(v) for v in r.split(',')] for r in data])
    areas = np.prod(np.array([np.abs(np.subtract.outer(input[:,n],input[:,n])+1) for n in range(2)]),0)
    return np.max(areas)

@hf.timing
def part2(data):
    input = np.array([[int(v) for v in r.split(',')] for r in data])
    areas = np.prod(np.array([np.abs(np.subtract.outer(input[:,n],input[:,n]))+1 for n in range(2)]),0)

    xs = np.unique(input[:,0])
    ys = np.unique(input[:,1])

    # Compressing coordinates
    x_to_comp = {x: i for i, x in enumerate(xs)}
    y_to_comp = {y: i for i, y in enumerate(ys)}
    compressed_input = np.array([
        [x_to_comp[x], y_to_comp[y]] for x, y in input
    ])
    # compressed_input = input
    max_x, max_y = np.max(compressed_input, 0)

    floor_map = np.zeros((max_y+2, max_x+2), dtype=int)
    floor_map[compressed_input[:,[1]], compressed_input[:,[0]]] = RED_TILE

    red_tiles_indices = compressed_input[:,[1,0]]
    red_tiles_indices_shifted = red_tiles_indices.take(range(1, len(red_tiles_indices)+1),
                                                       axis=0,
                                                       mode="wrap")
    # np.savetxt('floor_map_1.txt', np.where(floor_map == 1,"#"," "), delimiter='',fmt='%c')

    # Fill borders
    border_coords = []
    for start_tile, end_tile in zip(red_tiles_indices, red_tiles_indices_shifted):
        range_row = range(min(start_tile[0], end_tile[0]), max(start_tile[0], end_tile[0]+1))
        range_col = range(min(start_tile[1], end_tile[1]), max(start_tile[1], end_tile[1]+1))
        static_coord = range_row.start if len(range_row) == 1 else range_col.start 
        border_coords.extend(zip_longest(range_row, range_col, fillvalue=static_coord))
    border_rows, border_cols = list(zip(*border_coords))
    floor_map[border_rows, border_cols] = 2
    # np.savetxt('floor_map_2.txt', np.where(floor_map == 2,"#"," "), delimiter='',fmt='%c')

    # Fill outside
    num_rows, num_cols = floor_map.shape
    top = np.column_stack((np.full(num_cols+2, -1), np.arange(-1, num_cols+1)))
    bottom = np.column_stack((np.full(num_cols+2, num_rows), np.arange(-1, num_cols+1)))
    left = np.column_stack((np.arange(-1, num_rows+1), np.full(num_rows+2, -1)))
    right = np.column_stack((np.arange(-1, num_rows+1), np.full(num_rows+2, num_cols)))

    cells_to_fill = deque(np.vstack([top, bottom, left, right]))
    while(cells_to_fill):
        cell = cells_to_fill.popleft()
        if (-1 < cell[0] < num_rows) and (-1 < cell[1] < num_cols):
            floor_map[cell] = OUTSIDE
        surrounding_cells = [(cell[0]-1, cell[1]), (cell[0], cell[1]-1), (cell[0], cell[1]+1), (cell[0]+1, cell[1])]
        for cell_candidate in surrounding_cells:
            if ((-1 < cell_candidate[0] < num_rows) and
                (-1 < cell_candidate[1] < num_cols)  and
                (floor_map[cell_candidate] == UNKNOWN)):
                cells_to_fill.append(cell_candidate)
    # np.savetxt('floor_map_3.txt', np.where(floor_map == -1,"X"," "), delimiter='',fmt='%c')

    # Get areas of tiled rectangles
    red_green_tiles = np.where(floor_map < 0,1,0)
    red_green_tile_areas = np.zeros(areas.shape, dtype=bool)
    for tile_idx_1, (col1, row1) in enumerate(compressed_input[:,:]):
        for tile_idx_2, (col2, row2) in enumerate(compressed_input[:,:]):
            col_start, col_end = sorted([col1, col2])
            row_start, row_end = sorted([row1, row2])
            num_cols = col_end - col_start + 1
            num_rows = row_end - row_start + 1
            red_green_tile_areas[tile_idx_1, tile_idx_2] = np.sum(red_green_tiles[row_start:row_end+1,
                                                                                  col_start:col_end+1]) == 0
    # np.savetxt('red_green_tiles.txt', np.where(red_green_tiles,"X"," "), delimiter='',fmt='%c')
    # np.savetxt('red_green_tile_areas.txt', red_green_tile_areas, delimiter=',')

    return max(areas[red_green_tile_areas])

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

@pytest.fixture
def full_input():
    with open("inputDay09.txt") as f:
        input = [line.strip() for line in f.readlines()]
    return input

@pytest.mark.parametrize("test_input,expected", [(35, 35),
                                                 (26, 26)])
def test_help_function(test_input, expected):
    assert test_input == expected

def test_part1(input):
    assert part1(input) == 50

def test_part2(input):
    assert part2(input) == 24

def test_part2_fill():
    inp = ["1,1", "1,3", "4,3", "4,5", "1,5", "1,10", "3,10", "3,7", "5,7", "5,10", "10,10", "10,6", "7,6", "7,4", "10,4", "10,1", "7,1", "7,3", "5,3", "5,1"]
    assert part2(inp) == 30

def test_part1_full_input(full_input):
    assert part1(full_input) == 4740155680

def test_part2_full_input(full_input):
    assert part2(full_input) == 1543501936


## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    print(f"Part1 result: {part1(input.copy())}")
    print(f"Part2 result: {part2(input.copy())}")
