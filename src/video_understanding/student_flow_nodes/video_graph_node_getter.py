import functools

from . import compile_options
from .. import video_flow_graph
from ...flow import internal_graph_node


class _VideoGraphNodes:

    def __init__(self, video_fname: str):
        self.graph = video_flow_graph.VideoFlowGraph(makeviz=False, dry_run=True)
        self.graph.persist_graph_for(video_fname)

    @property
    def current_highlights_node(self) -> internal_graph_node.AddedNode:

        match compile_options.COMPILATION_TYPE:
            case compile_options.COMPILATION_TYPE.STUDENT_HIRING:
                highlights_node = self.graph.highlights_student_hiring
            case compile_options.COMPILATION_TYPE.STUDENT_RESUME:
                highlights_node = self.graph.highlights_student_resume
            case compile_options.COMPILATION_TYPE.TEACHER_HIRING:
                highlights_node = self.graph.highlights_teacher_hiring
            case _:
                raise ValueError(
                    f"Unknown compilation type: {compile_options.COMPILATION_TYPE}"
                )
        return highlights_node


# This is thread-safe. The maxsize (roughly) should be the number of threads needed.
@functools.lru_cache(maxsize=10)
def get_video_graph_nodes(video_fname: str) -> _VideoGraphNodes:
    return _VideoGraphNodes(video_fname)
