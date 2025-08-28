import abc
import inspect

from . import type_util

from typing import Any


# Naming suggestion:
# Name your module as agent-nouns (ending with -er). E.g. speaker_diarizer.
# Name your class as the name of the module, e.g. SpeakerDiarizer.
# Name instance as verb with _node suffix,
#   e.g. speaker_diarize_node = graph.add_node(...).
class ProcessNode(abc.ABC):
    @abc.abstractmethod
    def process(self, **kwargs) -> Any:
        """Override to define a processor  node. Type-checked at runtime.

        The output must be json-compatible. This is to ensure it can be stored
        in a human-readable format.

        If a large output is needed, for example a video, it can be stored in a
        file or blob, and the file path should be returned.
        """
        pass

    @classmethod
    def name(cls) -> str:
        # Class method since name must be accessible without initialization.
        #
        # This default should be fine in most cases. Override if the class name
        # changes, or there is collision.
        return cls.__name__

    def validate_args(self, kwargs) -> None:
        """Validates kwargs against the overridden process()'s signature.

        Normally the default implementation should suffice. Override for
        specialized nodes.

        Raises:
            TypeError: If validation fails.
        """
        sig = inspect.signature(self.process)
        bound = sig.bind(**kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            expected_type = sig.parameters[name].annotation
            if not type_util.matches(value, expected_type):
                raise TypeError(f"Type not matched: {value!r} is not {expected_type}")

    def finalize(self) -> None:
        """Use this to release resources, such as background servers."""
        pass
