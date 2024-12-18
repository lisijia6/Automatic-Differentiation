#!/usr/bin/env python3
# File       : plot_graph.py

import numpy as np
from AutoDiff import Forward, graphvis
"""
PLEASE install the `AutoDiff` package following the instruction on README using the following command before you run the code:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff
# for this computation graph plotting feature, you need to install graphviz on your machine and pip install graphviz
```
This script demonstrates how to plot computation graph using AutoDiff
The computation graph is generated in the folder AutoDiff/graphvis/
The computation graph is named computationGraph.png
"""

x = [3, 4, 5]
def f2(x1, x2, x3):
   	return [np.sin(x1)-x2-3/x3, np.cos(x2)*x1/x3]
g2 = Forward(f2, *x)
graphvis.generate_graph(x, g2)
