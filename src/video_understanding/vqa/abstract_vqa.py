import abc


class AbstractVqa(abc.ABC):
    @abc.abstractmethod
    def __init__(self, video_path: str):
        pass

    @abc.abstractmethod
    def ask(self, time: float, question: str) -> str:
        raise NotImplementedError
