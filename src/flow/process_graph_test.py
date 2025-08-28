import json
import os
import tempfile
import time
import unittest

from . import internal_graph_node
from . import process_graph
from . import process_node

from typing import Any, override

# pyright: reportPrivateUsage=false


# Example node for testing.
class SumInt(process_node.ProcessNode):
    def __init__(self) -> None:
        self.process_call_count = 0

    @override
    def process(self, a: int, b: int) -> int:
        self.process_call_count += 1
        return a + b


class Inc(process_node.ProcessNode):
    def __init__(self, how_much: int) -> None:
        self._how_much_inc = how_much

    @override
    def process(self, a: int) -> int:
        return a + self._how_much_inc


def _decrement_graph(
    num_nodes: int,
) -> tuple[process_graph.ProcessGraph, list[internal_graph_node.AddedNode]]:
    class TestNode(process_node.ProcessNode):
        @override
        def process(self, a: int) -> int:
            result = a - 1
            # Fail if we reach 0.
            # So if we have 10 nodes and start with 10, failure will occur.
            if result <= 0:
                raise ValueError("Test error")
            return result

    graph = process_graph.ProcessGraph()

    n1 = graph.add_constant_node(1, name="test_constant", value=0)
    nodes = [n1]
    for i in range(2, num_nodes + 1):
        nodes.append(graph.add_node(i, TestNode, {"a": nodes[-1]}))

    return graph, nodes


def _results_without_meta(graph: process_graph.ProcessGraph) -> dict[str, Any]:
    # Remove things like output_ts and time for test comparisons.
    result = graph._results_dict.copy()
    for result_item in result.values():
        del result_item["meta"]
    return result


class TestProcessGraph(unittest.TestCase):
    def test_simple_graph_execution(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": node1}, version=2)

        # Run node2 and check its result.
        result_node2 = graph.run_upto([node2])
        self.assertEqual(result_node2, 6)  # (1+2) + (1+2) = 3 + 3 = 6

        # Check the results_dict.
        expected_results_dict = {
            1: {"name": "SumInt", "output": 3, "version": 0},
            2: {"name": "SumInt", "output": 6, "version": 2},
        }
        self.assertEqual(_results_without_meta(graph), expected_results_dict)

    def test_compute_only_once(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(
            2, SumInt, {"a": node1, "b": 3}, invalidate_before=time.time() + 60 * 600
        )
        sum_node1: SumInt = node1._node  # pyright: ignore
        sum_node2: SumInt = node2._node  # pyright: ignore

        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 1)
        self.assertEqual(sum_node2.process_call_count, 1)

        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 1)
        self.assertEqual(sum_node2.process_call_count, 2)

        graph.reset()
        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 2)
        self.assertEqual(sum_node2.process_call_count, 3)

        graph.release_resources()
        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 2)
        self.assertEqual(sum_node2.process_call_count, 3)

    def test_dependency_updated(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": 3})
        sum_node1: SumInt = node1._node  # pyright: ignore
        sum_node2: SumInt = node2._node  # pyright: ignore

        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 1)
        self.assertEqual(sum_node2.process_call_count, 1)

        # Normally, no recomputation is done.
        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 1)
        self.assertEqual(sum_node2.process_call_count, 1)

        # But if a dependency is updated, update the node.
        assert node2.result_timestamp is not None
        node1.result_timestamp = node2.result_timestamp + 1
        self.assertEqual(graph.run_upto([node2]), 6)
        self.assertEqual(sum_node1.process_call_count, 1)
        self.assertEqual(sum_node2.process_call_count, 2)

    def test_change_inputs_and_rerun(self):
        graph = process_graph.ProcessGraph()
        node = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        self.assertEqual(graph.run_upto([node]), 3)

    def test_duplicate_node_id(self):
        graph = process_graph.ProcessGraph()
        _ = graph.add_node(1, SumInt, {"a": 1, "b": [2]})
        with self.assertRaises(ValueError):
            _ = graph.add_node(1, SumInt, {"a": 1, "b": [2]})

    def test_type_validation(self):
        graph = process_graph.ProcessGraph()
        node = graph.add_node(1, SumInt, {"a": 1, "b": [2]})
        with self.assertRaises(TypeError):
            graph.run_upto([node])

        node = graph.add_node(2, SumInt, {"a": [1], "b": 2})
        with self.assertRaises(TypeError):
            graph.run_upto([node])

    def test_constant_node(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_constant_node(1, name="test_constant", value="hello")
        self.assertEqual(graph.run_upto([node1]), "hello")

        graph.reset()
        node1.set_value("world")
        self.assertEqual(graph.run_upto([node1]), "world")

    def test_persistence_partial(self):
        graph = process_graph.ProcessGraph()
        const = graph.add_constant_node(1, name="test_constant", value=2)
        graph.run_upto([const])

        results_dict = json.loads(json.dumps(graph._results_dict))

        graph = process_graph.ProcessGraph()
        const = graph.add_constant_node(1, name="test_constant", value=2)
        sum_node = graph.add_node(2, SumInt, {"a": 1, "b": const})
        graph._load_results_dict(results_dict)
        self.assertEqual(graph.run_upto([sum_node]), 3)

    def test_persistence(self):
        def make_graph():
            graph = process_graph.ProcessGraph()
            node1 = graph.add_node(2, SumInt, {"a": 1, "b": 2})
            node2 = graph.add_node(3, SumInt, {"a": node1, "b": node1})
            return graph, node2

        graph, final_node = make_graph()
        result = graph.run_upto([final_node])
        results_dict = graph._results_dict

        # Results dict should survive jsonification.
        results_dict_reloaded = json.loads(json.dumps(results_dict))

        # Remake the graph, load, and test.
        graph, final_node = make_graph()
        graph._load_results_dict(results_dict_reloaded)
        self.assertEqual(results_dict, graph._results_dict)
        self.assertEqual(result, graph.run_upto([final_node]))
        # There should be no new computation when we called graph.run_upto([final_node]).
        self.assertEqual(final_node._node.process_call_count, 0)  # type: ignore

    def test_graph_structure(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": node1})
        node3 = graph.add_node(3, SumInt, {"a": node1, "b": node2})
        self.assertEqual(graph._dependencies, {1: set(), 2: {1}, 3: {1, 2}})

        self.assertEqual(graph._topological_sort([node3]), [node1, node2, node3])
        self.assertEqual(graph._topological_sort([node2]), [node1, node2])
        self.assertEqual(graph._topological_sort([node2, node3]), [node1, node2, node3])

    def test_node_with_init_args(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(
            1, Inc, constructor_kwargs=dict(how_much=20), inputs={"a": 5}
        )

        self.assertEqual(graph.run_upto([node1]), 25)

    def test_manual_override(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": 3})
        node3 = graph.add_node(3, SumInt, {"a": node2, "b": 4})
        self.assertEqual(graph.run_upto([node3]), 10)

        def override_fn(original_result, **kwargs):
            self.assertEqual(original_result, 6)
            self.assertEqual(kwargs, {"a": 3, "b": 3})
            return 7

        node2.manual_override_func = override_fn
        node3.reset()
        self.assertEqual(graph.run_upto([node3]), 11)

    def test_recompute_new_version(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_node(1, SumInt, {"a": 1, "b": 2})
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": node1})
        sum_node2: SumInt = node2._node  # pyright: ignore

        # Run node2 and check its result.
        result_node2 = graph.run_upto([node2])
        self.assertEqual(result_node2, 6)  # (1+2) + (1+2) = 3 + 3 = 6
        self.assertEqual(sum_node2.process_call_count, 1)

        # Simulate save and reload.
        node2.from_persist(node2.to_persist())
        # No recompute if version is same.
        result_node2 = graph.run_upto([node2])
        self.assertEqual(result_node2, 6)  # (1+2) + (1+2) = 3 + 3 = 6
        self.assertEqual(sum_node2.process_call_count, 1)

        # Simulate save and reload.
        node2.from_persist(node2.to_persist())
        # Recompute if version changed.
        node2.version = 1
        result_node2 = graph.run_upto([node2])
        self.assertEqual(result_node2, 6)  # (1+2) + (1+2) = 3 + 3 = 6
        self.assertEqual(sum_node2.process_call_count, 2)

    def test_batch_process_needs_persist(self):
        graph, nodes = _decrement_graph(num_nodes=10)

        def prep_fn(index, item):
            nodes[0].set("value", item)

        with self.assertRaisesRegex(ValueError, r"persist\(\) must be called"):
            graph.process_batch(
                batch_items=[11, 9, 5, 10],
                run_nodes=[nodes[-1]],
                prep_fn=prep_fn,
                release_resources_after=nodes,
            )

    def test_batch_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph, nodes = _decrement_graph(num_nodes=10)

            def prep_fn(index: int, item: Any) -> None:
                nodes[0].set("value", item)
                graph.persist(os.path.join(temp_dir, "persist" + str(index)))

            stats = graph.process_batch(
                batch_items=[10, 9, 21, 5],
                run_nodes=[nodes[-1]],
                prep_fn=prep_fn,
                release_resources_after=nodes,
            )

            # Verify one of the computations, after loading it with persist.
            graph.persist(os.path.join(temp_dir, "persist2"))
            self.assertEqual(
                [graph._results_dict[node.id]["output"] for node in nodes],
                [21, 20, 19, 18, 17, 16, 15, 14, 13, 12],
            )

        # Only two should succeed, 10 and 21.
        self.assertEqual(stats.completed, 2)

        # 9 and 5 should fail, since decreasing 10 times will make them lower than 0.
        self.assertEqual({x.item for x in stats.failures}, {9, 5})

    def test_batch_process_fail_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph, nodes = _decrement_graph(num_nodes=5)

            def prep_fn(index, item):
                nodes[0].set("value", item)
                graph.persist(os.path.join(temp_dir, "persist" + str(index)))

            # Values 9 and 5 will cause failure.
            with self.assertRaisesRegex(ValueError, r"Test error"):
                graph.process_batch(
                    batch_items=[11, 2, 1, 10],
                    run_nodes=[nodes[-1]],
                    prep_fn=prep_fn,
                    release_resources_after=nodes,
                    fault_tolerant=False,  # Make it fail immediately.
                )

    def test_batch_process_fail_fast_no_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph, nodes = _decrement_graph(num_nodes=5)

            def prep_fn(index, item):
                nodes[0].set("value", item)
                graph.persist(os.path.join(temp_dir, "persist" + str(index)))

            # Does not fail.
            graph.process_batch(
                batch_items=[11, 10],
                run_nodes=[nodes[-1]],
                prep_fn=prep_fn,
                release_resources_after=nodes,
                fault_tolerant=False,  # On error, do not continue.
            )

    def test_volatile(self):
        graph = process_graph.ProcessGraph()
        node1 = graph.add_constant_node(1, name="test_constant", value=2)
        node2 = graph.add_node(2, SumInt, {"a": node1, "b": node1})

        graph.run_upto([node2])

        sum_node2: SumInt = node2._node  # pyright: ignore
        self.assertEqual(node2.result, 4)
        self.assertEqual(sum_node2.process_call_count, 1)
        self.assertEqual(
            _results_without_meta(graph),
            {
                1: {"name": "test_constant", "output": 2, "version": 0},
                2: {"name": "SumInt", "output": 4, "version": 0},
            },
        )

        node1.set_value(3)
        graph.run_upto([node2])
        self.assertEqual(sum_node2.process_call_count, 1)
        self.assertEqual(node2.result, 4)
        self.assertEqual(
            _results_without_meta(graph),
            {
                # A volatile node will be always rerun.
                1: {"name": "test_constant", "output": 3, "version": 0},
                # But it will not trigger dependant nodes to update.
                2: {"name": "SumInt", "output": 4, "version": 0},
            },
        )

        # However if dependant node is recomputed for any other reason, it will use the new value of volatile node.
        node2.version = 1
        graph.run_upto([node2])
        self.assertEqual(sum_node2.process_call_count, 2)
        self.assertEqual(node2.result, 6)
        self.assertEqual(
            _results_without_meta(graph),
            {
                1: {"name": "test_constant", "output": 3, "version": 0},
                2: {"name": "SumInt", "output": 6, "version": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
