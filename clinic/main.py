from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml

app = FastAPI()
# Получение сессии базы данных с помощью SQLAlchemy
# Подключение к базе данных
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# Создание сессии базы данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Определение базовой модели для создания таблиц в базе данных
Base = declarative_base()

# Определение схемы базы данных (модель данных)
class Dog(Base):
    __tablename__ = "Dog"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    kind = Column(String, index=True)
    pk = Column(Integer, index=True)

# Создание таблиц в базе данных
Base.metadata.create_all(bind=engine)

# Определение эндпоинтов
@app.post("/dog/" )
def create_item(name: str, kind: str,pk:int):
    try:
        db = SessionLocal()
        dbDog = Dog(name=item.name, kind=item.kind , pk = item.pk )
        db.add(dbDog)
        db.commit()
        db.refresh(dbDog)
        return db_item
    except:
        return {"detail": [{"error": "Failed to create"}]}
@app.get("/dog/")
def read_items():
    db = SessionLocal()
    dogs = db.query(Dog).all()
    return dogs

@app.get("/dog/{pk}")
async def read_dog(pk: int):
    db = SessionLocal()
    dog = db.query(Dog).filter(Dog.id == pk).first()
    db.close()
    if dog is None:
        raise HTTPException(status_code=404, detail="Dog not found")
    return {"name": dog.name, "breed": dog.breed}
@app.post("/dog/{pk}")
async def post_dog(pk: int):
    db = SessionLocal()
    dog = db.query(Dog).filter(Dog.id == pk).first()
    db.close()
    if dog is None:
        raise HTTPException(status_code=404, detail="Dog not found")
    return {"name": dog.name, "breed": dog.breed}

@app.post("/")
def root_post():
    try:
        db = SessionLocal()
        result = ""
        with db:
            query = select([models.Dog]).order_by(models.Dog.id.desc()).limit(1)
            result = db.execute(query).fetchone()
        dbDog = Dog(name="Test", kind="Kind" , pk = result.pk)
        db.add(dbDog)
        db.commit()
        db.refresh(dbDog)
        return db_item
    except:
        return {"detail": [{"error": "Failed to create"}]}


@app.get("/")
def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
