# main.py
import os
import uuid
import re
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pydantic import BaseModel

# 数据库配置
from database import SessionLocal, engine, DetectionRecord, Base

# 响应模型
class DetectionRecordResponse(BaseModel):
    id: int
    image_path: str
    result_path: str
    detection_time: datetime
    fire_count: int
    smoke_count: int

    class Config:
        orm_mode = True


# 初始化FastAPI应用
app = FastAPI(
    title="火灾烟雾检测系统API",
    description="提供实时检测和历史记录查询服务",
    version="1.0.0"
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,#1212
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件配置
STATIC_DIR = "static"
os.makedirs(f"{STATIC_DIR}/uploads", exist_ok=True)
os.makedirs(f"{STATIC_DIR}/results", exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 加载YOLO模型
model = YOLO("models/best.pt")  # 替换为你的模型路径

# 数据库依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_db():
    # 创建数据库表
    Base.metadata.create_all(bind=engine)

# 工具函数
async def save_upload_file(file: UploadFile, save_path: str):
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

# 检测接口
@app.post("/detect/",
          response_model=dict,
          status_code=201,
          summary="上传检测图片",
          responses={
              201: {"description": "检测成功"},
              400: {"description": "无效文件类型"},
              500: {"description": "检测失败"}
          })
async def detect_fire_smoke(
        file: UploadFile = File(..., description="需要检测的图片文件（JPEG/PNG）"),
        db: Session = Depends(get_db)
):
    # 验证文件类型
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="仅支持JPEG/PNG格式图片"
        )

    # 初始化路径变量用于错误清理
    upload_path = None
    result_path = None

    try:
        # ================== 文件名处理 ==================
        original_name = re.sub(r'[^\w\u4e00-\u9fa5_.-]', '_', file.filename)  # 支持中文
        base_name, ext = os.path.splitext(original_name)
        unique_id = uuid.uuid4().hex[:8]
        saved_filename = f"{base_name}_{unique_id}{ext.lower()}"  # 统一小写扩展名

        # ================== 路径生成 ==================
        upload_dir = os.path.join(STATIC_DIR, "uploads")
        result_dir = os.path.join(STATIC_DIR, "results")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        upload_path = os.path.join(upload_dir, saved_filename)
        result_path = os.path.join(result_dir, saved_filename)

        # ================== 保存文件 ==================
        try:
            await save_upload_file(file, upload_path)
            if not os.path.exists(upload_path):
                raise HTTPException(500, "文件保存验证失败")
        except IOError as e:
            raise HTTPException(500, f"文件写入失败: {str(e)}")

        # ================== 模型推理 ==================
        try:
            results = model.predict(
                source=upload_path,
                imgsz=640,
                conf=0.25,
                save=False
            )
            if not results:
                raise ValueError("模型未返回结果")
            result = results[0]
        except Exception as model_error:
            raise HTTPException(500, f"推理错误: {str(model_error)}")

        # ================== 保存结果 ==================
        try:
            im_array = result.plot(line_width=2, font_size=10)
            im = Image.fromarray(im_array[..., ::-1])
            im.save(result_path)
            if not os.path.exists(result_path):
                raise RuntimeError("结果文件生成失败")
        except Exception as save_error:
            raise HTTPException(500, f"结果保存失败: {str(save_error)}")

        # ================== 数据库操作 ==================
        try:
            record = DetectionRecord(
                original_name=original_name,
                saved_name=saved_filename,
                image_path=f"uploads/{saved_filename}",  # 存储相对路径
                result_path=f"results/{saved_filename}",
                fire_count=sum(1 for cls in result.boxes.cls if cls == 0),
                smoke_count=sum(1 for cls in result.boxes.cls if cls == 1),
                detection_time=datetime.now()
            )
            db.add(record)
            db.commit()
            db.refresh(record)
        except SQLAlchemyError as db_error:
            db.rollback()
            raise HTTPException(500, f"数据库错误: {str(db_error)}")

        return {
            "status": "success",
            "result_url": f"/static/results/{saved_filename}",  # 修正返回路径
            "fire_count": record.fire_count,
            "smoke_count": record.smoke_count,
            "record_id": record.id
        }

    except HTTPException:
        raise  # 直接传递已处理的HTTP异常
    except Exception as e:
        # ================== 清理残留文件 ==================
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
        if result_path and os.path.exists(result_path):
            os.remove(result_path)
        raise HTTPException(
            status_code=500,
            detail=f"检测失败: {str(e)}"
        )

# 历史记录接口
@app.get("/records/",
         response_model=List[DetectionRecordResponse],
         summary="获取检测历史",
         description="分页查询带过滤条件的检测记录")
def get_records(
        page: int = Query(1, ge=1, description="页码（从1开始）"),
        per_page: int = Query(6, le=100, description="每页数量（最大100）"),  # 根据截图显示6条修改默认值
        start_time: Optional[datetime] = Query(None, description="开始时间（ISO格式）"),
        end_time: Optional[datetime] = Query(None, description="结束时间（ISO格式）"),
        min_fire: Optional[int] = Query(None, ge=0, description="最小火灾数量"),
        db: Session = Depends(get_db)
):
    # 构建基础查询
    query = db.query(DetectionRecord)

    # 时间范围过滤
    if start_time or end_time:
        if start_time and end_time:
            query = query.filter(DetectionRecord.detection_time.between(start_time, end_time))
        elif start_time:
            query = query.filter(DetectionRecord.detection_time >= start_time)
        else:
            query = query.filter(DetectionRecord.detection_time <= end_time)

    # 火灾数量过滤
    if min_fire is not None:
        query = query.filter(DetectionRecord.fire_count >= min_fire)

    # 执行分页查询
    records = query.order_by(desc(DetectionRecord.detection_time)) \
        .offset((page - 1) * per_page) \
        .limit(per_page) \
        .all()

    # 转换路径为可访问的URL
    for record in records:
        record.image_path = f"/static/uploads/{os.path.basename(record.image_path)}"
        record.result_path = f"/static/results/{os.path.basename(record.result_path)}"
    return records

# 健康检查接口
@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}