#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest
import numpy as np

@hf.timing
def part1(data):
    input = np.array([[int(v) for v in r.split(',')] for r in data])
    areas = np.prod(np.array([np.abs(np.subtract.outer(input[:,n],input[:,n])+1) for n in range(2)]),0)
    return np.max(areas)

@hf.timing
def part2(data):
    return 0

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

## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    print(f"Part1 result: {part1(input.copy())}")
    print(f"Part2 result: {part2(input.copy())}")
