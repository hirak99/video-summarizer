import dataclasses
import datetime
import gc
import json
import logging
import os
import time
from typing import Any, Callable, Protocol, Type, TypeVar, Generic

from . import graph_algorithms
from . import process_node

# If True, will not initialize node constructors, will not compute, and will not save.
DRY_RUN = False


# Use this to manually override any node computation results.
#
# Notes:
# - The funciton should take the form -
#     def override(original_result, **kwargs) -> Any
#   If you spell out kwargs, pyright will complain. Also you will then need to keep
#   track of the signature including inputs you don't want to use.
# - It should return a result even if there is no change due to override.
# - Changes to overrides are not detected automatically. You need to invalidate_before.
# - To change files, you should create a new file with the override.
class _ManualOverrideFuncT(Protocol):
    def __call__(self, original_result: Any, **kwargs: Any) -> Any: ...


@dataclasses.dataclass
class AddedNode:
    """Returned by graph.add_node(). Do not instantiate manually."""

    id: int
    version: int
    node_class: Type[process_node.ProcessNode]
    constructor_args: dict[str, Any]
    inputs: dict[str, "AddedNode | Any"]
    # Any results before this will be ignored and recomputed.
    # Tip: Use `date +%s` to get the current time in seconds since epoch.
    invalidate_before: float
    on_result: Callable[["AddedNode"], None]

    # Sometimes you will need to override the computed results. You may use this
    # hook to do that. See doc on the type for usage suggestions.
    manual_override_func: _ManualOverrideFuncT | None

    # Set to true if any dependant nodes used this, and an override changed the value.
    _was_overridden_in_dependancy: bool = False

    # Note: lru_cache does not work since this is not hashable.
    _result: Any = None
    _result_version: int = 0
    _result_timestamp: float | None = None
    _time: float | None = None

    # Lazy init this only if needed.
    # Prevents node to be created if existing data is loaded.
    _lazy_node: process_node.ProcessNode | None = None

    def has_result(self) -> bool:
        return self._result_timestamp is not None

    # Control access of this; because accessing this will initialize the node.
    @property
    def _node(self) -> process_node.ProcessNode:
        if self._lazy_node is None:
            self._lazy_node = self.node_class(**self.constructor_args)
        return self._lazy_node

    @property
    def name(self) -> str:
        return self.node_class.name()

    def reset(self):
        self._result = None
        self._result_timestamp = None
        self._time = None

    def release_resources(self):
        """Clears all loaded models."""
        if self._lazy_node is not None:
            logging.info(f"Releasing resources for node {self.id}: {self.name}")
            self._lazy_node.finalize()
            self._lazy_node = None
        self.reset()
        gc.collect()

    def set(self, arg_name: str, value: Any):
        if arg_name not in self.inputs:
            raise ValueError(f"Argument was not found in add_node(...): {arg_name}")
        self.inputs[arg_name] = value

    def to_persist(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "output": self._result,
            "meta": {
                "output_ts": self._result_timestamp,
                "time": self._time,
            },
        }
        if self._was_overridden_in_dependancy:
            assert isinstance(result["meta"], dict)
            result["meta"]["overriden"] = True
        result["version"] = self._result_version
        return result

    def from_persist(self, saved_result: dict[str, Any]) -> None:
        if saved_result["name"] != self.name:
            logging.warning(
                f"Node {self.id} has changed from {saved_result['name']!r} to {self.name!r}. "
                "Attempting to load anyway."
            )
        self._result = saved_result["output"]
        # Note that the saved version is not the .version, it is ._result_version.
        if "version" in saved_result:
            self._result_version = saved_result["version"]
        # TODO: Obsolete, delete this.
        if "output_ts" in saved_result:
            self._result_timestamp = saved_result["output_ts"]
        if "meta" in saved_result:
            self._result_timestamp = saved_result["meta"]["output_ts"]
            self._time = saved_result["meta"]["time"]
            if "overriden" in saved_result["meta"]:
                self._was_overridden_in_dependancy = saved_result["meta"]["overriden"]

    @property
    def _overriden_result(self) -> Any:
        """Returns result with any manual overrides if applicable."""
        if self.manual_override_func is None:
            return self._result
        new_result = self.manual_override_func(self._result, **self._filled_in_inputs)
        if self._result != new_result:
            self._was_overridden_in_dependancy = True
            logging.warning(
                f"Overriding has changed the output of {self.id} ({self.name})"
            )
        else:
            logging.info(
                f"Overriding has not changed the output of {self.id} ({self.name})"
            )
        return new_result

    @property
    def _filled_in_inputs(self) -> dict[str, Any]:
        """Converts any inputs which are nodes to their results."""
        kwargs = {}
        for input_name, input_val in self.inputs.items():
            if isinstance(input_val, AddedNode):
                # The following line would recurse dependencies.
                # kwargs[input_name] = input_val.internal_run()
                # However we manage dependencies in the graph.
                if not input_val.has_result():
                    raise ValueError(
                        f"Dependent node was not run: id={input_val.id} {input_val.name}"
                    )
                kwargs[input_name] = input_val._overriden_result
            else:
                kwargs[input_name] = input_val
        try:
            self._node.validate_args(kwargs)
        except TypeError as e:
            raise TypeError(
                f"Error validating arguments for node {self.id}: {self._node.name()} with {kwargs!r}"
            ) from e
        return kwargs

    def _refresh_result(self):
        kwargs = self._filled_in_inputs
        start = time.time()
        if DRY_RUN:
            self._result = None
        else:
            self._result = self._node.process(**kwargs)
        self._time = time.time() - start
        self._result_timestamp = datetime.datetime.now().timestamp()
        self._result_version = self.version
        self.on_result(self)

    def _needs_update(self) -> bool:
        if not self.has_result():
            logging.info(f"Needs update ({self.id}): {self.name} because no result")
            return True
        assert self._result_timestamp is not None  # Because self.has_result() is True.
        if self._result_version != self.version:
            logging.info(
                f"Needs update ({self.id}): {self.name} because version {self._result_version} < {self.version}"
            )
            return True
        if self._result_timestamp < self.invalidate_before:
            logging.info(
                f"Needs update ({self.id}): {self.name} because timestamp {self._result_timestamp} < {self.invalidate_before}"
            )
            return True

        # Update if any of the dependencies had a recent update.
        for input_name, input_val in self.inputs.items():
            del input_name  # Unused
            if isinstance(input_val, AddedNode):
                if (
                    input_val._result_timestamp is not None
                    and input_val._result_timestamp > self._result_timestamp
                ):
                    logging.info(
                        f"Needs update ({self.id}) {self.name} because dependency is newer:"
                        f" {input_val._result_timestamp} > {self._result_timestamp}"
                    )
                    return True
        return False

    def internal_run(self):
        if self._needs_update():
            logging.info(f"Updating node ({self.id}): {self.name}")
            self._refresh_result()
        else:
            logging.info(f"Returning precomputed for {self.id}: {self._result}")
        return self._result


# Generic type for process_batch arguments.
_BatchItemT = TypeVar("_BatchItemT")


@dataclasses.dataclass
class _BatchFailure(Generic[_BatchItemT]):
    item_index: int
    item: _BatchItemT
    failed_node: AddedNode
    exception: Exception


@dataclasses.dataclass
class _BatchStats(Generic[_BatchItemT]):
    completed: int
    # Contains the item_index, item, node, and the exception.
    failures: list[_BatchFailure[_BatchItemT]]


class ProcessGraph:
    def __init__(self):
        # Nodes with no dependents (used for running a subset of the graph).
        self._all_nodes: dict[int, AddedNode] = {}
        self._auto_save_path: str | None = None
        # Used only for visualizing graph. NodeInstance handles actual call
        # dependencies.
        self._dependencies: dict[int, set[int]] = {}

    def _save_to(self, path: str):
        if DRY_RUN:
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

    def _on_node_result(self, node: AddedNode):
        if self._auto_save_path is not None:
            self._save_to(self._auto_save_path)

    def add_node(
        self,
        id: int,
        node_class: Type[process_node.ProcessNode],
        inputs: dict[str, AddedNode | Any],
        version: int = 0,
        constructor_kwargs: dict[str, Any] | None = None,
        invalidate_before: float = 0,
        force: bool = False,
        manual_override_func: _ManualOverrideFuncT | None = None,
    ) -> AddedNode:
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
        node_instance = AddedNode(
            id=id,
            version=version,
            node_class=node_class,
            constructor_args=constructor_kwargs or {},
            inputs=inputs,
            on_result=self._on_node_result,
            invalidate_before=invalidate_before,
            manual_override_func=manual_override_func,
        )
        self._all_nodes[id] = node_instance
        # Update DAG.
        self._dependencies[id] = set()
        for val in inputs.values():
            if isinstance(val, AddedNode):
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

    def run_upto(self, final_nodes: list[AddedNode]) -> Any:
        last_result: Any = None
        for node in self._topological_sort(final_nodes):
            last_result = node.internal_run()
        return last_result

    def _topological_sort(
        self,
        final_nodes: list[AddedNode],
    ) -> list[AddedNode]:
        """Computes topological sort of all the nodes.

        This can be used to run a batch processing with 'depth-first' mode, to
        optimally use resources such as VRAM. See the README.md for more
        details.

        Args:
            starting_node: The node to be run finally. Only it and its
            dependencies will be sorted.
        """
        result: list[AddedNode] = []
        for node_id in graph_algorithms.topo_sort_subgraph(
            {x.id for x in final_nodes}, self._dependencies
        ):
            result.append(self._all_nodes[node_id])
        return result

    def _run_only(self, node: AddedNode) -> Any:
        return node.internal_run()

    def process_batch(
        self,
        *,
        batch_items: list[_BatchItemT],
        final_nodes: list[AddedNode],
        prep_fn: Callable[[int, _BatchItemT], None],
        post_fn: Callable[[int, _BatchItemT], None] | None = None,
        release_after_nodes: list[AddedNode] | None = None,
        fault_tolerant: bool = True,
    ) -> _BatchStats[_BatchItemT]:
        """Runs the nodes breath-first, for efficient resource managmement.

        Args:
            inputs: The list of items to process.
            final_nodes: The nodes which need evaluated. All dependant nodes will automatically be evaluated.
            prep_fn: Called before a node is run. This (1) should call graph.persist(FILE_BASED_ON_ITEM), and (2) should set constants.
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
