import dataclasses
import json
import logging
import os
import time

from . import graph_algorithms
from . import internal_graph_node
from . import process_node

from typing import Any, Callable, Generic, Type, TypeVar

# Generic type for process_batch arguments.
_BatchItemT = TypeVar("_BatchItemT")


@dataclasses.dataclass
class _BatchFailure(Generic[_BatchItemT]):
    """Information collected on items that fail during batch_process()."""

    item_index: int
    item: _BatchItemT
    failed_node: internal_graph_node.AddedNode
    exception: Exception


@dataclasses.dataclass
class _BatchStats(Generic[_BatchItemT]):
    """Returned by process_batch()."""

    # Number of items on which all nodes succeeded.
    completed: int
    # Contains the item_index, item, node, and the exception.
    failures: list[_BatchFailure[_BatchItemT]]


class ProcessGraph:
    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        # Nodes with no dependents (used for running a subset of the graph).
        self._all_nodes: dict[int, internal_graph_node.AddedNode] = {}
        self._auto_save_path: str | None = None
        # Used only for visualizing graph. NodeInstance handles actual call
        # dependencies.
        self._dependencies: dict[int, set[int]] = {}

    def _save_to(self, path: str):
        if self._dry_run:
            return
        logging.info(f"Saving graph state to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results_dict, f)

    def persist(self, path: str):
        """Sets a file where the results will be saved.

        This must be set before any computation (i.e. node.run()) is done.
        If the file exists, previous results will be loaded.
        """
        self.reset()
        try:
            with open(path) as f:
                self._load_results_dict(json.load(f))
        except FileNotFoundError:
            pass
        self._auto_save_path = path

    def _on_node_result(self, node: internal_graph_node.AddedNode):
        if self._auto_save_path is not None:
            self._save_to(self._auto_save_path)

    def add_node(
        self,
        id: int,
        node_class: Type[process_node.ProcessNode],
        inputs: dict[str, internal_graph_node.AddedNode | Any],
        version: int = 0,
        constructor_kwargs: dict[str, Any] | None = None,
        invalidate_before: float = 0,
        force: bool = False,
        manual_override_func: internal_graph_node.ManualOverrideFuncT | None = None,
    ) -> internal_graph_node.AddedNode:
        """Adds a processor node to the graph.

        Args:
            id: An integer identifying the node. Prefer counting up from 0. Do not re-use deleted node ids.
            node_class: Class of the node to add.
            inputs: The inputs to the node. If PipelineNode, then it is output of that node. Other values are passed as-is.
            version: Increment this when the node logic is changed, and it needs to be recomputed.
            constructor_kwargs: Passed directly to the node's class when it is instantiated.
            invalidate_before: Alternative to version, set this to a time when node logic was changed. Prefer version when possible.
            force: The node will be always recomputed. Use this sparingly, prefer other solutions when possible.
            manual_override_func: Change the output manually. Experimental - may become obsoleted.

        Returns:
            A representation of the node. It can be passed to other nodes as input.
        """
        if id in self._all_nodes:
            raise ValueError(f"Node id already added: {id}")

        if force:
            # Redo if done before 100 years from now.
            invalidate_before = time.time() + (100 * 365 * 24 * 60 * 60)

        # Create the underlying node and its instance.
        node_instance = internal_graph_node.AddedNode(
            id=id,
            version=version,
            node_class=node_class,
            constructor_args=constructor_kwargs or {},
            inputs=inputs,
            on_result=self._on_node_result,
            invalidate_before=invalidate_before,
            manual_override_func=manual_override_func,
            dry_run=self._dry_run,
        )
        self._all_nodes[id] = node_instance
        # Update DAG.
        self._dependencies[id] = set()
        for val in inputs.values():
            if isinstance(val, internal_graph_node.AddedNode):
                self._dependencies[id].add(val.id)
        # Return the instance so that it can be used as inputs to other nodes.
        return node_instance

    def reset(self):
        """Clear cached information."""
        for node_instance in self._all_nodes.values():
            node_instance.reset()

    def release_resources(self):
        """Clear the node and free up memory."""
        for node_instance in self._all_nodes.values():
            node_instance.release_resources()

    @property
    def results_dict(self):
        """Returns current results from the graph."""
        results = {}
        for id, node_instance in self._all_nodes.items():
            if node_instance.has_result():
                results[id] = node_instance.to_persist()
        return results

    def _load_results_dict(self, results_dict: dict[int, dict[str, Any]]):
        loaded_count = 0
        for node_id_str, saved_result in results_dict.items():
            # Need int(node_id) since when converting back from json, keys become strings.
            node_id = int(node_id_str)
            if node_id not in self._all_nodes:
                # May be the computation till this far has not been done.
                continue
            node_instance = self._all_nodes[node_id]
            node_instance.from_persist(saved_result)
            loaded_count += 1
        logging.info(
            f"Loaded nodes from disk: {loaded_count} of {len(self._all_nodes)}"
        )

    def run_upto(self, final_nodes: list[internal_graph_node.AddedNode]) -> Any:
        last_result: Any = None
        for node in self._topological_sort(final_nodes):
            last_result = node.internal_run()
        return last_result

    def _topological_sort(
        self,
        final_nodes: list[internal_graph_node.AddedNode],
    ) -> list[internal_graph_node.AddedNode]:
        """Computes topological sort of all the nodes.

        This can be used to run a batch processing with 'depth-first' mode, to
        optimally use resources such as VRAM. See the README.md for more
        details.

        Args:
            starting_node: The node to be run finally. Only it and its
            dependencies will be sorted.
        """
        result: list[internal_graph_node.AddedNode] = []
        for node_id in graph_algorithms.topo_sort_subgraph(
            {x.id for x in final_nodes}, self._dependencies
        ):
            result.append(self._all_nodes[node_id])
        return result

    def _run_only(self, node: internal_graph_node.AddedNode) -> Any:
        return node.internal_run()

    def process_batch(
        self,
        *,
        batch_items: list[_BatchItemT],
        final_nodes: list[internal_graph_node.AddedNode],
        prep_fn: Callable[[int, _BatchItemT], None],
        post_fn: Callable[[int, _BatchItemT], None] | None = None,
        release_after_nodes: list[internal_graph_node.AddedNode] | None = None,
        # TODO: Only if needed, replace or add faults_per_node_allowed.
        fault_tolerant: bool = True,
    ) -> _BatchStats[_BatchItemT]:
        """Runs the nodes breath-first, for efficient resource managmement.

        Args:
            inputs: The list of items to process.
            final_nodes: The nodes which need evaluated. All dependant nodes will automatically be evaluated.
            prep_fn: Called before a node is run. This (1) must call graph.persist(FILE_BASED_ON_ITEM), and (2) should set constants.
            post_fn: Called after a node is run.
            release_after_nodes: Nodes which are used for heavy computation. When these are used, resources are freed up.
            fault_tolerant: If True, will continue other items and summarize errors at the end. If False, will stop immediately if a node execution fails.
        """

        batch_result = _BatchStats[_BatchItemT](completed=0, failures=[])

        # Indexed by the items.
        indices_with_errors: set[int] = set()

        nodes_to_run = self._topological_sort(final_nodes)

        for node_index, node in enumerate(nodes_to_run):
            is_last_node = node_index == len(nodes_to_run) - 1

            for item_index, item in enumerate(batch_items):
                if item_index in indices_with_errors:
                    logging.info(f"Skipping {item_index} due to previous error.")
                    continue

                # It's essential to have the results persist for breath-first running.
                self._auto_save_path = None
                prep_fn(item_index, item)
                if self._auto_save_path is None:
                    raise ValueError(f"persist() must be called in prep_fn")

                try:
                    self._run_only(node)
                except Exception as exc:
                    if not fault_tolerant:
                        raise exc
                    logging.warning(
                        f"Error processing {item!r} for node {node.id}: {node.name}. Error: {exc}"
                    )
                    indices_with_errors.add(item_index)
                    # batch_result.failures.append((index, item, node, exc))
                    batch_result.failures.append(
                        _BatchFailure(item_index, item, node, exc)
                    )
                    continue

                if is_last_node:
                    batch_result.completed += 1

                if post_fn is not None:
                    post_fn(item_index, item)

            if release_after_nodes is not None:
                if node in release_after_nodes:
                    # Free up resources for the next batch after using heavy nodes.
                    self.release_resources()

        # Explicitly release resources after processing all inputs.
        self.release_resources()

        return batch_result
