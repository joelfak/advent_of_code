#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest, logging, re
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

@dataclass
class MachineConfiguration:
    wanted_state: list
    button_wiring: list
    joltage: str

def find_smallest_number_of_presses(mc: MachineConfiguration):
    logger.debug(f"wanted_state: {mc.wanted_state}")
    logger.debug(f"wiring: {len(mc.button_wiring)}, {mc.button_wiring}")
    for num_presses in range(1, len(mc.button_wiring) + 1):
        for combination in combinations(mc.button_wiring, num_presses):
            tested_state = mc.wanted_state + [button for buttons in combination for button in buttons]
            if sum([v % 2 for v in Counter(tested_state).values()]) == 0:
                logger.debug(f"Num presses: {num_presses}, combination: {combination}")
                return num_presses

def find_smallest_number_of_presses_for_joltage(mc: MachineConfiguration):
    logger.debug(f"\njoltage: {mc.joltage}")
    goal_counter = Counter({i:j for i, j in enumerate(mc.joltage)})
    logger.debug(f"goal_counter: {goal_counter}")
    logger.debug(f"wiring: {len(mc.button_wiring)}, {mc.button_wiring}")

    min_button_presses = max(mc.joltage)
    max_button_presses = sum(mc.joltage) + 1
    # logger.debug(f"min: {min_button_presses}, max: {max_button_presses}")
    for num_presses in range(min_button_presses, max_button_presses):
        logger.debug(f"Testing {num_presses} presses")
        for combination in combinations_with_replacement(mc.button_wiring, num_presses):
            # logger.debug(f"Test combination: {combination}")
            total_button_presses = Counter([button for buttons in combination for button in buttons])
            total_button_presses.subtract(goal_counter)
            if all((v == 0 for v in total_button_presses.values())):
                logger.debug(f"Solution found: {Counter((tuple(comb) for comb in combination))}")
                return num_presses
    logger.debug("Solution not found")

def parse_input(data) -> MachineConfiguration:
    pattern = re.compile(r"\[(?P<wanted>[\.#]+)\] (?P<wiring>(?:\([\d,]+\) )+)\{(?P<joltage>[\d,]+)\}")
    machine_configurations = []

    for line in data:
        result = pattern.search(line)
        # logger.debug(f"\nline: {line}")
        if result:
            wanted_state = [n for n, v in enumerate(result["wanted"]) if v == "#"]
            button_wiring = [list(map(int, button_wiring.strip(")").strip("(").split(",")))
                             for button_wiring in result["wiring"].split()]
            joltage = list(map(int, result["joltage"].split(",")))
            # logger.debug(f"wanted_state: {wanted_state}")
            # logger.debug(f"button_wiring: {button_wiring}")
            # logger.debug(f"joltage: {joltage}")
            machine_configurations.append(MachineConfiguration(wanted_state, button_wiring, joltage))
    return machine_configurations

@hf.timing
def part1(data):
    machine_configurations = parse_input(data)
    return sum((find_smallest_number_of_presses(config) for config in machine_configurations))

@hf.timing
def part2(data):
    machine_configurations = parse_input(data)
    return sum((find_smallest_number_of_presses_for_joltage(config) for config in machine_configurations))

## Unit tests ########################################################

@pytest.fixture
def input():
    return ["[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}",
            "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}",
            "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}]",]

@pytest.mark.parametrize("test_input,expected", [(MachineConfiguration([1, 2],[[3],[1,3],[2],[2,3],[0,2],[0,1]],[]), 2),
                                                 (MachineConfiguration([3],[(0,2,3,4),(2,3),(0,4),(0,1,2),(1,2,3,4)],[]), 3),
                                                 (MachineConfiguration([1,2,3,5],[(0,1,2,3,4),(0,3,4),(0,1,2,4,5),(1,2)],[]), 2)
                                                 ])
def test_find_smallest_number_of_presses(test_input, expected):
    assert find_smallest_number_of_presses(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(MachineConfiguration([],[[3],[1,3],[2],[2,3],[0,2],[0,1]],[3,5,4,7]), 10),
                                                 (MachineConfiguration([],[(0,2,3,4),(2,3),(0,4),(0,1,2),(1,2,3,4)],[7,5,12,7,2]), 12),
                                                 (MachineConfiguration([],[(0,1,2,3,4),(0,3,4),(0,1,2,4,5),(1,2)],[10,11,11,5,10,5]), 11)
                                                 ])
def test_find_smallest_number_of_presses_for_joltage(test_input, expected):
    assert find_smallest_number_of_presses_for_joltage(test_input) == expected

def test_part1(input):
    assert part1(input) == 7

def test_part2(input):
    assert part2(input) == 33

## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    # print(f"Part1 result: {part1(input.copy())}")
    print(f"Part2 result: {part2(input.copy())}")
