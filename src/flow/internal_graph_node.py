# Internal class.
# Okay for typing purposes, but DO NOT INSTANTIATE manually.
#
# If you are looking for abstraction to define a node, see process_node.py.
import dataclasses
import datetime
import gc
import logging
import time

from . import process_node

from typing import Any, Callable, Protocol, Type


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
class ManualOverrideFuncT(Protocol):
    def __call__(self, original_result: Any, **kwargs: Any) -> Any: ...


@dataclasses.dataclass
class AddedNode:
    """Returned by graph.add_node(). Do not instantiate manually."""

    id: int
    version: int | str
    node_class: Type[process_node.ProcessNode]
    constructor_args: dict[str, Any]
    inputs: dict[str, "AddedNode | Any"]
    # Any results before this will be ignored and recomputed.
    # Tip: Use `date +%s` to get the current time in seconds since epoch.
    invalidate_before: float
    on_result: Callable[["AddedNode", bool], None]

    # Sometimes you will need to override the computed results. You may use this
    # hook to do that. See doc on the type for usage suggestions.
    manual_override_func: ManualOverrideFuncT | None

    # If True, will not initialize node constructors, will not compute, and will not save.
    dry_run: bool

    # Passive nodes do not trigger update for dependencies even if value changes.
    #
    # They are useful for maintenance constants such as file paths, since you
    # may not want to redo extensive pipeline computation if you merely move
    # your source files.
    passive: bool

    # Which arg will be set if .set() is called without arg name.
    default_arg_to_set: str | None

    # Set to true if any dependant nodes used this, and an override changed the value.
    _was_overridden_in_dependancy: bool = False

    # Note: Do not set the following values outside the core flow architecture classes.
    result: Any = None
    _result_version: int | str = 0
    # When was this result computed.
    result_timestamp: float | None = None
    # How long did the computation take.
    _time: float | None = None

    # Lazy init this only if needed.
    # Prevents node to be created if existing data is loaded.
    _lazy_node: process_node.ProcessNode | None = None

    def has_result(self) -> bool:
        return self.result_timestamp is not None

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
        self.result = None
        self.result_timestamp = None
        self._time = None

    def release_resources(self):
        """Clears all loaded models."""
        if self._lazy_node is not None:
            logging.info(f"Releasing resources for node {self.id}: {self.name}")
            self._lazy_node.finalize()
            self._lazy_node = None
        self.reset()
        gc.collect()

    def set_value(self, value: Any) -> None:
        if self.default_arg_to_set is None:
            raise ValueError("No default arg to set")
        self.set(self.default_arg_to_set, value)

    def set(self, arg: str, value: Any) -> None:
        if arg not in self.inputs:
            raise ValueError(f"Argument was not found in add_node(...): {arg}")
        self.inputs[arg] = value

    def to_persist(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "output": self.result,
            "meta": {
                "output_ts": self.result_timestamp,
                "time": self._time,
            },
        }
        assert isinstance(result["meta"], dict)
        if self.passive:
            result["meta"]["passive"] = True
        if self._was_overridden_in_dependancy:
            result["meta"]["overriden"] = True
        result["version"] = self._result_version
        return result

    def from_persist(self, saved_result: dict[str, Any]) -> None:
        if saved_result["name"] != self.name:
            logging.warning(
                f"Node {self.id} has changed from {saved_result['name']!r} to {self.name!r}. "
                "Attempting to load anyway."
            )
        self.result = saved_result["output"]
        # Note that the saved version is not the .version, it is ._result_version.
        if "version" in saved_result:
            self._result_version = saved_result["version"]
        # TODO: Obsolete, delete this.
        if "output_ts" in saved_result:
            self.result_timestamp = saved_result["output_ts"]
        if "meta" in saved_result:
            self.result_timestamp = saved_result["meta"]["output_ts"]
            self._time = saved_result["meta"]["time"]
            if "overriden" in saved_result["meta"]:
                self._was_overridden_in_dependancy = saved_result["meta"]["overriden"]

    @property
    def _overriden_result(self) -> Any:
        """Returns result with any manual overrides if applicable."""
        if self.manual_override_func is None:
            return self.result
        new_result = self.manual_override_func(self.result, **self._filled_in_inputs)
        if self.result != new_result:
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
        prev_result = self.result
        if self.dry_run:
            self.result = None
        else:
            self.result = self._node.process(**kwargs)
        self._time = time.time() - start
        self.result_timestamp = datetime.datetime.now().timestamp()
        self._result_version = self.version
        self.on_result(self, prev_result != self.result)

    def _needs_update(self) -> bool:
        if self.passive:
            return True
        if not self.has_result():
            logging.info(f"Needs update ({self.id}): {self.name} because no result")
            return True
        assert self.result_timestamp is not None  # Because self.has_result() is True.
        if self._result_version != self.version:
            logging.info(
                f"Needs update ({self.id}): {self.name} because version {self._result_version!r} != {self.version!r}"
            )
            return True
        if self.result_timestamp < self.invalidate_before:
            logging.info(
                f"Needs update ({self.id}): {self.name} because timestamp {self.result_timestamp} < {self.invalidate_before}"
            )
            return True

        # Update if any of the dependencies had a recent update.
        for input_name, input_val in self.inputs.items():
            del input_name  # Unused
            if isinstance(input_val, AddedNode):
                if (
                    not input_val.passive
                    and input_val.result_timestamp is not None
                    and input_val.result_timestamp > self.result_timestamp
                ):
                    logging.info(
                        f"Needs update ({self.id}) {self.name} because dependency is newer:"
                        f" {input_val.result_timestamp} > {self.result_timestamp}"
                    )
                    return True
        return False

    def internal_run(self):
        if self._needs_update():
            logging.info(f"Updating node ({self.id}): {self.name}")
            self._refresh_result()
        else:
            logging.info(f"Returning precomputed for {self.id}: {self.result}")
        return self.result
