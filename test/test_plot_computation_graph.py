import numpy as np
from AutoDiff import Forward
from AutoDiff.graphvis.plot_computation_graph import generate_graph
from pathlib import Path

class TestComputationGraph:
    """This is a class that tests the computation graph plotting feature.
    """

    def test_plotting(self):
        """Test for generating computation graph
        """
        x = [3, 4, 5]
        def f2(x1, x2, x3):
            return [np.sin(x1)-x2-3/x3, np.cos(x2)*x1/x3]
        g2 = Forward(f2, *x)
        generate_graph(x, g2)
        my_file = Path("./computationGraph")
        assert my_file.exists()
