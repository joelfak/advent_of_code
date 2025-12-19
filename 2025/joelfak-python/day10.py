#!/usr/bin/env python3

from __future__ import annotations
import helpfunctions as hf
import sys, pytest, logging, re, math
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
    joltage: tuple

def find_smallest_number_of_presses(mc: MachineConfiguration):
    logger.debug(f"wanted_state: {mc.wanted_state}")
    logger.debug(f"wiring: {len(mc.button_wiring)}, {mc.button_wiring}")
    for num_presses in range(1, len(mc.button_wiring) + 1):
        for combination in combinations(mc.button_wiring, num_presses):
            tested_state = mc.wanted_state + [button for buttons in combination for button in buttons]
            if sum([v % 2 for v in Counter(tested_state).values()]) == 0:
                logger.debug(f"Num presses: {num_presses}, combination: {combination}")
                return num_presses

@dataclass
class Node:
    button_press: tuple
    parent: Node | None
    current_joltage: tuple
    f: float
    g: float
    h: float
    valid: bool

    def __init__(self, button_press:tuple, joltage_spec: list, parent: Node | None):
        self.button_press = button_press
        self.parent = parent
        if parent:
            self.current_joltage = tuple(sum(pair) for pair in zip(parent.current_joltage, button_press))
            self.g = parent.g + 1
        else:
            self.current_joltage = button_press
            self.g = 0
        self.valid = True
        joltage_remaining = tuple(a-b for a, b in zip(joltage_spec, self.current_joltage))
        if any((j < 0 for j in joltage_remaining)):
            self.valid = False
        self.h = math.sqrt(sum(jr**2 for jr in joltage_remaining))
        # self.h = max(joltage_remaining)
        # self.h = sum(a*b for a,b in zip(sorted(joltage_remaining), range(1, len(joltage_remaining)+1)))
        self.f = self.g + self.h

def find_smallest_number_of_presses_for_joltage(mc: MachineConfiguration):
    logger.debug(f"\njoltage: {mc.joltage}")
    # goal_counter = Counter({i:j for i, j in enumerate(mc.joltage)})
    # logger.debug(f"goal_counter: {goal_counter}")
    logger.debug(f"wiring: {len(mc.button_wiring)}, {mc.button_wiring}")    

    start_node = Node(tuple([0] * len(mc.joltage)),mc.joltage, None)
    open_nodes = {start_node.current_joltage: start_node}
    closed_nodes = {}

    searched_node_counter = 0
    while open_nodes:
        # logger.debug([j for j, n in open_nodes.items()])
        current_joltage, current_node = min(open_nodes.items(), key=lambda n: n[1].f)
        print(f"searched: {searched_node_counter}, current joltage: {sum(current_joltage)}{current_joltage}, g: {current_node.g:.3f}, h: {current_node.h:.3f}, f: {current_node.f:.3f}, button_press: {current_node.button_press}")
        del open_nodes[current_joltage]
        if current_node.current_joltage == mc.joltage:
            # logger.debug(f"Found path with {current_node.g} steps:")
            s = str(f"\n{current_node.button_press}")
            node = current_node
            while node := node.parent:
                if node.parent:
                    s += str(f"\n{node.button_press}")
            # logger.debug(s)
            return current_node.g
        closed_nodes[current_joltage] = current_node
        for button_wiring in mc.button_wiring:
            button_press = tuple(1 if v in button_wiring else 0 for v in range(len(mc.joltage)))
            next_node = Node(button_press, mc.joltage, current_node)
            if next_node.valid and next_node.current_joltage not in closed_nodes:
                if next_node.current_joltage in open_nodes:
                    if next_node.g < open_nodes[next_node.current_joltage].g:
                        open_nodes[next_node.current_joltage] = next_node
                else:
                    open_nodes[next_node.current_joltage] = next_node

            searched_node_counter += 1
        pass
    logger.warning("Solution not found")
    return 0

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
            # max_button_combinations_per_joltage_index = {}
            # button_press_ranges = {(idx: (m))}
            # for idx in range(len(joltage)):
            #     max_button_combinations_per_joltage_index[idx] = (joltage[idx], [i for i, button in enumerate(button_wiring) if idx in button])

            machine_configurations.append(MachineConfiguration(wanted_state, button_wiring, tuple(joltage)))
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
    return [
            # "[###.#.] (1,3,5) (0,2,5) (1,3) (2,3,5) (0,2,4) (0,1,2,4) {44,25,63,29,33,40}",
            "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}",
            "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}",
            "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}]",
            ]

@pytest.mark.parametrize("test_input,expected", [(MachineConfiguration([1, 2],[[3],[1,3],[2],[2,3],[0,2],[0,1]],[]), 2),
                                                 (MachineConfiguration([3],[(0,2,3,4),(2,3),(0,4),(0,1,2),(1,2,3,4)],[]), 3),
                                                 (MachineConfiguration([1,2,3,5],[(0,1,2,3,4),(0,3,4),(0,1,2,4,5),(1,2)],[]), 2)
                                                 ])
def test_find_smallest_number_of_presses(test_input, expected):
    assert find_smallest_number_of_presses(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(MachineConfiguration([],[[3],[1,3],[2],[2,3],[0,2],[0,1]],(3,5,4,7)), 10),
                                                 (MachineConfiguration([],[(0,2,3,4),(2,3),(0,4),(0,1,2),(1,2,3,4)],(7,5,12,7,2)), 12),
                                                 (MachineConfiguration([],[(0,1,2,3,4),(0,3,4),(0,1,2,4,5),(1,2)],(10,11,11,5,10,5)), 11)
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
