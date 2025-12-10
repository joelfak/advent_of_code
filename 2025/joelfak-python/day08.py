#!/usr/bin/env python3

import helpfunctions as hf
import sys, pytest
import numpy as np

class Circuit:
    def __init__(self, junction_boxes, connection):
        self.junction_boxes = {*junction_boxes}
        self.connections = {connection}

    def __repr__(self):
        return f"[{len(self.junction_boxes)} junction boxes, {len(self.connections)} connections]"

@hf.timing
def part1(data, num_connections = None):
    if not num_connections:
        num_connections = (len(data) ** 2 - len(data)) // 2

    input = np.array([[int(v) for v in r.split(',')] for r in data])
    distances = np.sqrt(np.sum(np.array([np.square(np.subtract.outer(input[:,n],input[:,n])) for n in range(3)]),0))
    distances_flattened = distances.ravel()

    one_way_dist_mask = np.array([[True if i > j else False
                                   for i in range(distances.shape[1])]
                                    for j in range(distances.shape[0])])
    one_way_dist_mask_flattened = one_way_dist_mask.ravel()
    one_way_distances_flattened = distances_flattened[one_way_dist_mask_flattened]
    partition_indices_in_mask = np.argpartition(one_way_distances_flattened, num_connections-1)[:num_connections]
    one_way_distances_indices_flat = np.nonzero(one_way_dist_mask_flattened)[0]
    flat_indices = one_way_distances_indices_flat[partition_indices_in_mask]
    ordered_min_distances = np.argsort(distances_flattened[flat_indices])
    flat_indices_ordered = flat_indices[ordered_min_distances]
    distances_flattened[flat_indices_ordered]
    
    rows, cols = np.unravel_index(flat_indices_ordered, distances.shape)

    circuits = []

    for row, col in zip(rows, cols):
        junction_box_a = tuple(input[row,:])
        junction_box_b = tuple(input[col,:])
        junction_box_found = False
        found_circuit_a = None
        found_circuit_b = None
        for circuit in circuits:
            if (junction_box_a in set(circuit.junction_boxes)) or (junction_box_b in set(circuit.junction_boxes)):
                junction_box_found = True
                if (junction_box_a in set(circuit.junction_boxes)) and (junction_box_b in set(circuit.junction_boxes)):
                    break

                if junction_box_a not in set(circuit.junction_boxes):
                    circuit.junction_boxes.add(junction_box_a)
                if junction_box_b not in set(circuit.junction_boxes):
                    circuit.junction_boxes.add(junction_box_b)
                circuit.connections.add((junction_box_a, junction_box_b))

                if not found_circuit_a:
                    found_circuit_a = circuit
                else:
                    found_circuit_b = circuit

        if found_circuit_a and found_circuit_b:
            found_circuit_a.junction_boxes.update(found_circuit_b.junction_boxes)
            found_circuit_a.connections.update(found_circuit_b.connections)
            circuits.remove(found_circuit_b)

        if not junction_box_found:
            circuits.append(Circuit([junction_box_a, junction_box_b], (junction_box_a, junction_box_b)))

        if len(circuits) == 1 and len(circuits[0].junction_boxes) == len(data):
            print("All nodes connected")

            return junction_box_a[0] * junction_box_b[0]

    three_largest_size_graph_sizes = np.array([len(c.junction_boxes) for c  in sorted(circuits, key=lambda c: len(c.junction_boxes), reverse=True)[:3]])

    return np.product(three_largest_size_graph_sizes)

@hf.timing
def part2(data):
    return part1(data)

## Unit tests ########################################################

@pytest.fixture
def input():
    return ["162,817,812",
            "57,618,57",
            "906,360,560",
            "592,479,940",
            "352,342,300",
            "466,668,158",
            "542,29,236",
            "431,825,988",
            "739,650,466",
            "52,470,668",
            "216,146,977",
            "819,987,18",
            "117,168,530",
            "805,96,715",
            "346,949,466",
            "970,615,88",
            "941,993,340",
            "862,61,35",
            "984,92,344",
            "425,690,689"]

@pytest.mark.parametrize("test_input,expected", [(35, 35),
                                                 (26, 26)])
def test_help_function(test_input, expected):
    assert test_input == expected

def test_part1(input):
    assert part1(input, 10) == 40 

def test_part2(input):
    assert part2(input) == 25272

## Main ########################################################

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        input = [line.strip() for line in f.readlines()]

    print("Advent of code day X")
    print(f"Part1 result: {part1(input.copy(), 1000)}")
    print(f"Part2 result: {part2(input.copy())}")
