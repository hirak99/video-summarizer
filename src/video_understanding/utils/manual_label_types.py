import pydantic

from typing import Literal


class BoxLabel(pydantic.BaseModel):
    # Unique type identifier for Pydantic to load this from JSON.
    annotation_type: Literal["Box"] = "Box"

    start: float
    end: float
    x: float
    y: float
    width: float
    height: float

    def model_dump_rounded(self):
        d = self.model_dump()
        # Drop extra long decimal points.
        d["x"] = round(self.x, 2)
        d["y"] = round(self.y, 2)
        d["width"] = round(self.width, 2)
        d["height"] = round(self.height, 2)
        return d


class AnnotationProps(pydantic.BaseModel):
    # Following comes from the UI.
    id: str
    name: str
    label: BoxLabel


class UserAnnotation(pydantic.BaseModel):
    annotations: list[AnnotationProps]


class AllAnnotationsV2(pydantic.BaseModel):
    # Annotation data format.
    format: Literal["v2"] = "v2"
    # Annotations by all workspaces.
    # If the user does not have a workspace defined, the key will default to username.
    # The key "by_user" is legacy name, better understood as "by_workspace".
    by_user: dict[str, UserAnnotation]

    @classmethod
    def load(cls, v2_file: str) -> "AllAnnotationsV2":
        with open(v2_file, "r") as f:
            return cls.model_validate_json(f.read())

    def save(self, v2_file: str):
        with open(v2_file, "w") as f:
            f.write(self.model_dump_json(indent=2))
