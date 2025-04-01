# main.py
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
    Path,  # 新增
    Response  # 新增
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 新增配置项
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 创建数据库表（在启动时）
Base.metadata.create_all(bind=engine)

# OAuth2方案
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


# 新增的Pydantic模型
class UserCreate(BaseModel):
    username: str
    password: str


class UserInDB(UserCreate):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


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
            hashed_password=hashed_password
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
    CORSMiddleware,  # 1212
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
    return {"access_token": access_token, "token_type": "bearer"}


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

        # 修改返回的图片路径为完整URL
        return {
            "status": "success",
            "image_url": f"http://localhost:8000/static/uploads/{saved_filename}",
            "result_url": f"http://localhost:8000/static/results/{saved_filename}",
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
        # 删除文件
        upload_file = os.path.join(STATIC_DIR, record.image_path)
        result_file = os.path.join(STATIC_DIR, record.result_path)
        if os.path.exists(upload_file):
            os.remove(upload_file)
        if os.path.exists(result_file):
            os.remove(result_file)

        # 删除记录
        db.delete(record)
        db.commit()
    except SQLAlchemyError as e:  # 使用已导入的异常类型
        db.rollback()
        raise HTTPException(status_code=500, detail=f"数据库错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件操作失败: {str(e)}")

    return Response(status_code=204)


@app.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    """
    用户退出登录
    - 前端需要清除本地存储的token
    - 服务端使token失效（当前实现无状态，实际需要客户端主动清除）
    """
    return {"status": "200", "message": "退出成功"}
