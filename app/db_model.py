from pydantic import BaseModel

# 주차장 생성용
class LotCreate(BaseModel):
    name: str

class Lots(BaseModel):
    id: int
    name: str
    cnt: int

class Lot(BaseModel):
    id: int
    name: str
    cnt: int
