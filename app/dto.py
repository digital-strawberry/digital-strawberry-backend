from pydantic import BaseModel


class Entity(BaseModel):
    x: int
    y: int
    width: int
    height: int


class StrawberryPredictions(BaseModel):
    url: str
    health: int
    entities: list[Entity]
