# import functools

from .. import video_flow_graph
from ...flow import internal_graph_node
from ..video_flow_nodes import video_flow_types


class _VideoGraphNodes:

    def __init__(
        self,
        program: video_flow_types.ProgramType,
        movie_type: video_flow_types.CompilationType,
        video_fname: str,
    ):
        self._movie_type = movie_type

        self.graph = video_flow_graph.VideoFlowGraph(
            program=program, makeviz=False, dry_run=True
        )
        self.graph.persist_graph_for(video_fname)

    @property
    def current_highlights_node(self) -> internal_graph_node.AddedNode:

        match self._movie_type:
            case self._movie_type.STUDENT_HIRING:
                highlights_node = self.graph.highlights_student_hiring
            case self._movie_type.STUDENT_RESUME:
                highlights_node = self.graph.highlights_student_resume
            case self._movie_type.TEACHER_HIRING:
                highlights_node = self.graph.highlights_teacher_hiring
            case self._movie_type.FTP_HIGHLIGHTS:
                highlights_node = self.graph.highlights_ftp
            case _:
                raise ValueError(f"Unknown compilation type: {self._movie_type}")
        return highlights_node


# This is thread-safe. The maxsize (roughly) should be the number of threads needed.
# @functools.lru_cache(maxsize=10)
def get_video_graph_nodes(
    program: video_flow_types.ProgramType,
    movie_type: video_flow_types.CompilationType,
    video_fname: str,
) -> _VideoGraphNodes:
    return _VideoGraphNodes(
        program=program, movie_type=movie_type, video_fname=video_fname
    )
