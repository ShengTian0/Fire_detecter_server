from database import User, get_db
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from PIL import Image
# 数据库配置
from database import SessionLocal, engine, DetectionRecord, Base

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Query,
    Path,
    Response,
    Form,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from ultralytics import YOLO
# 登录需要
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import timedelta
from pydantic import BaseModel

from sqlalchemy import JSON
from database import Notification

import json

from realtime_detection import router as realtime_detection_router





STATIC_DIR = "static"
PREDICT_DIR = os.path.join(STATIC_DIR, "results", "predict")  # 统一结果目录
os.makedirs(PREDICT_DIR, exist_ok=True)

# ================== 安全配置 ==================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# 创建数据库表（在启动时）
Base.metadata.create_all(bind=engine)

# OAuth2方案
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# 新增的Pydantic模型
class UserCreate(BaseModel):
    username: str
    password: str
    role: str


class UserInDB(UserCreate):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


class TokenData(BaseModel):
    username: str | None = None

class UserResponse(BaseModel):
    id: int
    username: str
    role: str




# 通知请求模型
class NotificationRequest(BaseModel):
    location: str
    time: str
    result: dict

class NotificationResponse(BaseModel):
    id: int
    location: str
    time: datetime
    result: dict

class Config:
    orm_mode = True



# 工具函数
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# 新增依赖项
async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


# 响应模型
class DetectionRecordResponse(BaseModel):
    id: int
    image_path: str
    result_path: str
    detection_time: datetime
    fire_count: int
    smoke_count: int

    class Config:
        from_attributes = True


# 密码上下文初始化（放在文件顶部其他配置之后）

def init_admin_account(db: Session):
    """
    初始化管理员账户
    """
    # 检查是否存在admin用户
    admin_user = db.query(User).filter(User.username == "admin").first()

    if not admin_user:
        # 生成安全哈希
        hashed_password = pwd_context.hash("admin123")
        new_user = User(
            username="admin",
            hashed_password=hashed_password,
            role='admin'
        )
        db.add(new_user)
        db.commit()
        print("[初始化] 管理员账户已创建")
    else:
        print("[初始化] 管理员账户已存在")


# 初始化FastAPI应用
app = FastAPI(
    title="火灾烟雾检测系统API",
    description="提供实时检测和历史记录查询服务",
    version="1.0.0"
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173","http://127.0.0.1:8000"], # Add your frontend origin here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["*"]
)

# 静态文件配置
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/static/results/predict", StaticFiles(directory=PREDICT_DIR), name="predict_results")

#vue3打包入fastapi中
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount("/dist", StaticFiles(directory=os.path.join(BASE_DIR, 'FireFinder_Server/dist')), name="dist")
app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, 'FireFinder_Server/dist/assets')), name="assets")

# 加载YOLO模型
model = YOLO("models/Fire&Smoke.pt")  # 替换为你的模型路径


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

    # 初始化管理员
    db = SessionLocal()
    try:
        init_admin_account(db)
    finally:
        db.close()


# 工具函数
async def save_upload_file(file: UploadFile, save_path: str):
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)


@app.get("/")
def main():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist', 'index.html')
    html_content = ''
    with open(html_path) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# 新增登录路由
@app.post("/login", response_model=Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role
    }
# 新增测试受保护路由
@app.get("/users/me")
async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]  # ✅ 确认算法一致
        )
        # 增加调试日志
        print(f"[DEBUG] 解析Token内容: {payload}")

        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="无效凭证")

        user = get_user(db, username=username)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")

        return user
    except JWTError as e:
        # 增加错误日志
        print(f"[JWT ERROR] {str(e)}")
        raise HTTPException(status_code=401, detail="Token验证失败")


# 检测接口


@app.post("/detect/", response_model=dict, status_code=201)
async def detect_fire_smoke(
        file: UploadFile = File(...),
        model: str = Form("Fire&Smoke"),
        confidence: float = Form(0.5),
        db: Session = Depends(get_db)
):
    # 文件类型验证
    if file.content_type not in ["image/jpeg", "image/png", "video/mp4"]:
        raise HTTPException(400, "仅支持JPEG/PNG/MP4格式")

    upload_path = None
    try:
        # 生成唯一文件名
        original_name = re.sub(r'[^\w\u4e00-\u9fa5_.-]', '_', file.filename)
        base_name, ext = os.path.splitext(original_name)
        unique_id = uuid.uuid4().hex[:8]
        saved_filename = f"{base_name}_{unique_id}{ext.lower()}"

        # 保存上传文件
        upload_dir = os.path.join(STATIC_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, saved_filename)
        await save_upload_file(file, upload_path)

        # 加载模型
        yolo_model = YOLO(f"models/{model}.pt")

        # 统一结果路径配置
        result_dir = PREDICT_DIR  # static/results/predict
        result_video_dir = os.path.join(STATIC_DIR, "results")

        if file.content_type == "video/mp4":
            # 视频处理（关键参数配置）
            results = yolo_model.predict(
                source=upload_path,
                imgsz=640,
                conf=confidence,
                save=True,
                project=result_video_dir,  # 指定到目标目录
                name="",  # 禁止生成子目录
                exist_ok=True,
                save_dir=result_video_dir  # 显式指定保存路径
            )

            # 获取生成的文件路径
            result_filename = os.path.basename(results[0].path).replace(".mp4", ".avi")
            result_path = os.path.join(result_dir, result_filename)

            # 统计检测结果
            fire_count = sum(1 for r in results for cls in r.boxes.cls if cls == 0)
            smoke_count = sum(1 for r in results for cls in r.boxes.cls if cls == 1)
        else:
            # 图片处理
            results = yolo_model.predict(upload_path, conf=confidence)
            result = results[0]

            # 生成结果文件名
            result_filename = f"{base_name}_detected_{unique_id}.jpg"
            result_path = os.path.join(result_dir, result_filename)

            # 保存检测结果
            Image.fromarray(result.plot()[..., ::-1]).save(result_path)
            fire_count = sum(1 for cls in result.boxes.cls if cls == 0)
            smoke_count = sum(1 for cls in result.boxes.cls if cls == 1)

        # 数据库记录（存储相对路径）
        record = DetectionRecord(
            original_name=original_name,
            saved_name=saved_filename,
            image_path=f"uploads/{saved_filename}",
            result_path=f"results/predict/{result_filename}",
            fire_count=fire_count,
            smoke_count=smoke_count,
            detection_time=datetime.now()
        )
        db.add(record)
        db.commit()

        return {
            "status": "success",
            "uploaded_url": f"http://localhost:8000/static/uploads/{saved_filename}",
            "result_url": f"http://localhost:8000/static/results/predict/{result_filename}",
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "record_id": record.id
        }

    except Exception as e:
        # 清理残留文件
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(500, f"检测失败: {str(e)}")

# 历史记录接口
@app.get("/records/", response_model=List[DetectionRecordResponse])
def get_records(
        page: int = Query(1, ge=1),
        per_page: int = Query(6, le=100),
        start_time: Optional[datetime] = Query(None),
        end_time: Optional[datetime] = Query(None),
        min_fire: Optional[int] = Query(None, ge=0),
        db: Session = Depends(get_db)
):
    query = db.query(DetectionRecord)

    # 时间过滤
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

    # 分页查询
    records = query.order_by(desc(DetectionRecord.detection_time)) \
        .offset((page - 1) * per_page) \
        .limit(per_page) \
        .all()

    # 转换路径格式
    for record in records:
        record.image_path = f"/static/{record.image_path}"
        record.result_path = f"/static/{record.result_path}"

    return records

app.include_router(realtime_detection_router)
# 删除接口
@app.delete("/records/{record_id}",
            status_code=204,
            summary="删除检测记录",
            responses={
                204: {"description": "删除成功"},
                404: {"description": "记录不存在"}
            })
def delete_record(
        record_id: int = Path(..., gt=0, description="记录ID"),
        db: Session = Depends(get_db)
):
    # 查询记录
    record = db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    try:
        # 获取静态文件绝对路径
        static_dir = os.path.abspath(STATIC_DIR)

        # 处理上传文件路径（格式：uploads/filename.jpg）
        upload_rel_path = record.image_path.lstrip("/")  # 去除可能存在的斜杠
        upload_file = os.path.join(static_dir, upload_rel_path)

        # 处理结果文件路径（格式：results/predict/filename.avi）
        result_rel_path = record.result_path.lstrip("/")  # 去除可能存在的斜杠
        result_file = os.path.join(static_dir, result_rel_path)

        # 打印调试信息
        print(f"[DEBUG] 待删除文件路径：")
        print(f"Upload: {upload_file} ({'存在' if os.path.exists(upload_file) else '不存在'})")
        print(f"Result: {result_file} ({'存在' if os.path.exists(result_file) else '不存在'})")

        # 删除文件
        if os.path.isfile(upload_file):
            os.remove(upload_file)
        if os.path.isfile(result_file):
            os.remove(result_file)

        # 删除记录
        db.delete(record)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"数据库错误: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件操作失败: {str(e)}. 请检查以下路径：\n"
                   f"上传文件: {upload_file}\n"
                   f"结果文件: {result_file}"
        )

    return Response(status_code=204)


@app.get("/users", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users


# Update user information
@app.put("/users/{user_id}", response_model=UserCreate)
async def update_user(user_id: int, user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.username = user.username
    db_user.hashed_password = get_password_hash(user.password)
    db_user.role = user.role

    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/notify", status_code=200)
async def send_notification(notification: NotificationRequest, db: Session = Depends(get_db)):
    try:
        # 解析时间字符串
        notification_time = datetime.fromisoformat(notification.time)

        # 序列化结果为JSON字符串
        result_json = json.dumps(notification.result)

        # 保存通知到数据库
        new_notification = Notification(
            location=notification.location,
            time=notification_time,
            result=result_json
        )
        db.add(new_notification)
        db.commit()
        db.refresh(new_notification)
        return {"status": "success", "message": "通知发送成功"}
    except Exception as e:
        # 打印错误日志
        print(f"通知发送失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"通知发送失败: {str(e)}")


@app.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
        page: int = Query(1, ge=1, description="页码（从1开始）"),
        per_page: int = Query(20, le=100, description="每页数量（最大100）"),
        db: Session = Depends(get_db)
):
    offset = (page - 1) * per_page
    notifications = db.query(Notification).offset(offset).limit(per_page).all()

    # Deserialize the JSON string to a dictionary
    for notification in notifications:
        notification.result = json.loads(notification.result)

    return notifications


@app.delete("/notifications/{notification_id}", status_code=204)
async def delete_notification(notification_id: int, db: Session = Depends(get_db)):
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="通知不存在")

    db.delete(notification)
    db.commit()
    return Response(status_code=204)

@app.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    """
    用户退出登录
    - 前端需要清除本地存储的token
    - 服务端使token失效（当前实现无状态，实际需要客户端主动清除）
    """
    return {"status": "200", "message": "退出成功"}
