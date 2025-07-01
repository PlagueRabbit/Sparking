# main.py (cnt = 현재 주차된 차량 수, capacity = 총 공간 수)
from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.staticfiles import StaticFiles
from app.yolo_nas import YOLO
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select, insert, update, delete, func
import base64
import shutil
import os
import json
from pydantic import BaseModel
from app.db_model import *
from app.api_communication import api_patch
from app.file_control import save_file


class LotCreate(BaseModel):
    name: str


# 예시로 주차장별 공간 수 정의
LOT_CAPACITY = {
    1: 50,
    2: 60,
    3: 50,
    4: 40,
    5: 60
}


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

trackers = {}  # 주차장별 YOLO 인스턴스 분리
engine = create_engine('sqlite:///spark.db')
metadata = MetaData()
parkinglots = Table('users', metadata,
              Column('id', Integer, primary_key=True),
              Column('name', String),
              Column('image', String, default="NULL"),
              Column('cnt', Integer, default=0))
metadata.create_all(engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트 포트
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 정적 파일 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 라즈베리파이에서 보내는 요청
yolo = YOLO(include_truck=False, confidence_threshold=0.3)


@app.patch("/photo/{idx}")
async def upload_and_detect(idx: int, file: UploadFile = File(...)):
    input_path = STATIC_DIR / f"uploaded{idx}.jpg"
    output_path = STATIC_DIR / f"predicted{idx}.jpg"
    json_path = STATIC_DIR / f"parking_{idx}.json"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if idx not in trackers:
        trackers[idx] = YOLO()
    yolo = trackers[idx]

    result = yolo.detect_and_track(str(input_path), str(output_path))
    parked_count = result["total_cars"]

    with engine.connect() as conn:
        conn.execute(update(parkinglots)
                     .where(parkinglots.c.id == idx)
                     .values(cnt=parked_count))
        conn.commit()

    total_spaces = LOT_CAPACITY.get(idx, 100)  # 없으면 기본 100
    available_spaces = max(0, total_spaces - parked_count)



    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "available_spaces": available_spaces,
            "long_parked": result["long_parked"],
            "total_spaces": total_spaces
        }, f, indent=2, ensure_ascii=False)

    return {
        "message": "Image and data processed",
        "total_cars": parked_count,
        "long_parked": result["long_parked"],
        "saved_image": output_path.name
    }


#@app.patch("/photo/{idx}")
#async def detection(file: UploadFile, idx: int):
#    try:
#        path = await save_file(file.file)
#        car = await yolo.count_car(path, idx)
#        with engine.connect() as conn:
#            with open(f'predicted{idx}.jpg', 'rb') as image_file:
#                encoded_image = base64.b64encode(image_file.read())
#                update_stmt = (
#                update(parkinglots)
#                .where(parkinglots.c.id == idx)
#                .values(image=encoded_image, cnt=car))
#            result = conn.execute(update_stmt)
#            conn.commit()
#
#            if result.rowcount == 0:
#                raise HTTPException(status_code=404, detail="Parking lot not found")
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#
#    return {"car_count": car, "message": "Detection and update successful"}

# 아래부터 클라이언트 통신
@app.get("/ParkingLots/", response_model=List[Lot])
async def read_lots():
    with engine.connect() as conn:
        result = conn.execute(select(parkinglots))
        parking_list = result.fetchall()
        return [Lot(id=row[0], name=row[1], image=row[2], cnt = row[3]) for row in parking_list]


@app.get("/ParkingLots/{lot_id}", response_model=Lot)
async def read_lot(lot_id: int):
    with engine.connect() as conn:
        result = conn.execute(select(parkinglots).where(parkinglots.c.id == lot_id))
        lot = result.fetchall()
        if lot:
            return lot[0]
        else:
            raise HTTPException(status_code=404, detail="User not found")


@app.post("/ParkingLots/", response_model=Lots)
async def create_lot(data: LotCreate):
    with engine.connect() as conn:
        result = conn.execute(insert(parkinglots).values(name=data.name))
        conn.commit()
        new_user_id = result.lastrowid
        row = conn.execute(select(parkinglots).where(parkinglots.c.id == new_user_id)).first()
        if row[3] == "NULL":
            row = (row[0], row[1], row[2], 0)  # cnt 값을 0으로 대체
        return Lots(id=row[0], name=row[1], cnt=row[3])



@app.delete("/ParkingLots/{lot_id}")
async def delete_lot(lot_id: int):
    with engine.connect() as conn:
        result = conn.execute(delete(parkinglots).where(parkinglots.c.id == lot_id))
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="주차장 없음")
    return {"message": f"{lot_id}번 주차장 삭제됨"}

#@app.post("/ParkingLots/", response_model=Lots)
#async def create_lot(new_name: str):
#    with engine.connect() as conn:
#        result = conn.execute(insert(parkinglots).values(name=new_name))
#        conn.commit()
#        new_user_id = result.lastrowid

        # 방금 만든 데이터 다시 조회해서 반환
#        row = conn.execute(select(parkinglots).where(parkinglots.c.id == new_user_id)).first()
#        return Lots(id=row[0], name=row[1], cnt=row[3])