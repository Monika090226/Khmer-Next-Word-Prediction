from sqlalchemy import Column, Integer, String, Float, Text, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())


class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    input_text = Column(Text)
    predicted_word = Column(String)
    score = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())