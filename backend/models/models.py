from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Smartlog(Base):
    __tablename__ = "tap_duty_device_log"
    __table_args__ = {"schema": "smartlog"}  # Set schema

    log_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    log_time = Column(TIMESTAMP, nullable=False)
    log_mode = Column(String, nullable=False)
