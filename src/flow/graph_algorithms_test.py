import unittest

from . import graph_algorithms

# Access private methods for tes.
# pyright: reportPrivateUsage=false


class TestGraphAlgorithms(unittest.TestCase):
    def test_graph1(self):
        graph = {
            6: {4, 5},
            5: {2},
            4: {2},
            3: {2},
            2: {1},
        }
        self.assertEqual(
            graph_algorithms._get_dependencies({6}, graph), {1, 2, 4, 5, 6}
        )
        self.assertEqual(graph_algorithms._get_dependencies({4}, graph), {1, 2, 4})

        self.assertEqual(graph_algorithms._topo_sort(graph), [1, 2, 3, 4, 5, 6])

        self.assertEqual(
            graph_algorithms.topo_sort_subgraph({6}, graph), [1, 2, 4, 5, 6]
        )
        self.assertEqual(graph_algorithms.topo_sort_subgraph({4}, graph), [1, 2, 4])
        self.assertEqual(graph_algorithms.topo_sort_subgraph({3}, graph), [1, 2, 3])
        self.assertEqual(
            graph_algorithms.topo_sort_subgraph({3, 4}, graph), [1, 2, 3, 4]
        )
