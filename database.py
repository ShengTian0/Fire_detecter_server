from datetime import datetime

from sqlalchemy import Column, Integer, String
from sqlalchemy import create_engine, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL 数据库配置
MYSQL_USER = "root"
MYSQL_PASSWORD = "031127"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "fire_detection"

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 声明基类
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(String(50))
    is_active= Column(Integer, default=0)  # 1表示激活，0表示未激活


# 定义检测记录模型
class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    original_name = Column(String(255), nullable=False)  # 原始文件名
    saved_name = Column(String(255), nullable=False)  # 存储的文件名
    # 修改现有字段为相对路径
    image_path = Column(String(255), nullable=False)  # 改为：uploads/文件名
    result_path = Column(String(255), nullable=False)  # 改为：results/文件名
    detection_time = Column(DateTime, default=datetime.now)
    fire_count = Column(Integer, default=0)
    smoke_count = Column(Integer, default=0)


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    location = Column(String(255), nullable=False)
    time = Column(DateTime, nullable=False)
    result = Column(String(255), nullable=False)  # Store result as JSON string


# 创建数据库表（如果不存在）
Base.metadata.create_all(bind=engine)
