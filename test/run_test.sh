#!/usr/bin/env bash

tests=(
    test_node.py # unit test
    test_rnode.py # unit test
    test_forward.py # integration test
    test_reverse.py # integration test
    test_optimization.py # integration test
    test_plot_computation_graph.py # integration test
    test_forward.py # regression test
    test_node.py # regression test
    test_rnode.py # regression test
)
export PYTHONPATH="$(pwd -P)/../AutoDiff":${PYTHONPATH}
python3 -m pytest ${tests[@]}