#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest, logging, re
from itertools import combinations
from collections import Counter

logger = logging.getLogger(__name__)

def find_smallest_number_of_presses(wanted_state, button_wiring):
    logger.debug(f"wanted_state: {wanted_state}")
    logger.debug(f"wiring: {len(button_wiring)}, {button_wiring}")
    for num_presses in range(1, len(button_wiring) + 1):
        for combination in combinations(button_wiring, num_presses):
            tested_state = wanted_state + [button for buttons in combination for button in buttons]
            if sum([v % 2 for v in Counter(tested_state).values()]) == 0:
                logger.debug(f"Num presses: {num_presses}, combination: {combination}")
                return num_presses

@hf.timing
def part1(data):
    pattern = re.compile(r"\[(?P<wanted>[\.#]+)\] (?P<wiring>(?:\([\d,]+\) )+)\{(?P<joltage>[\d,]+)\}")
    num_presses = 0
    for line in data:
        result = pattern.search(line)
        logger.debug(f"\nline: {line}")
        if result:
            wanted = [n for n, v in enumerate(result["wanted"]) if v == "#"]
            wiring = [list(map(int, button_wiring.strip(")").strip("(").split(",")))
                      for button_wiring in result["wiring"].split()]
            logger.debug(f"wanted: {wanted}")
            logger.debug(f"wiring: {wiring}")
            num_presses += find_smallest_number_of_presses(wanted, wiring)
    return num_presses

@hf.timing
def part2(data):
    return 0

## Unit tests ########################################################

@pytest.fixture
def input():
    return ["[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}",
            "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}",
            "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}]",]

@pytest.mark.parametrize("test_input,expected", [(([1, 2],[[3],[1,3],[2],[2,3],[0,2],[0,1]]), 2),
                                                 (([3],[(0,2,3,4),(2,3),(0,4),(0,1,2),(1,2,3,4)]), 3),
                                                 (([1,2,3,5],[(0,1,2,3,4),(0,3,4),(0,1,2,4,5),(1,2)]), 2)
                                                 ])
def test_find_smallest_number_of_presses(test_input, expected):
    assert find_smallest_number_of_presses(*test_input) == expected

def test_part1(input):
    assert part1(input) == 7

## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    print(f"Part1 result: {part1(input.copy())}")
    print(f"Part2 result: {part2(input.copy())}")
