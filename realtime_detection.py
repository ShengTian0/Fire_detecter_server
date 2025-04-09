import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, HTTPException,Body
from ultralytics import YOLO

# Create router
router = APIRouter()

# Load YOLO model
try:
    model = YOLO("models/Fire&Smoke.pt")
    print(f"🔥 模型加载成功 - 类别: {model.names}")  # 重要验证点
except Exception as e:
    raise RuntimeError(f"模型加载失败: {str(e)}")


@router.post("/detect-realtime")
async def detect_realtime(image: str = Body(..., embed=True)):
    """
    Detect objects in a base64-encoded image.
    """
    try:
        # Validate Base64 image format
        if not image.startswith("data:image"):
            raise ValueError("Invalid Base64 image format")

        # Decode Base64 image
        image_data = base64.b64decode(image.split(",")[1])
        try:
            image = Image.open(BytesIO(image_data))
        except UnidentifiedImageError:
            raise ValueError("Unrecognized image format")

        # Perform detection
        results = model.predict(image, conf=0.1)
        print(f"🖼️ 收到图像: {image.size} | 检测结果数: {len(results[0].boxes)}")  # 关键日志

        detections = []

        for result in results:
            for box in result.boxes:
                detections.append({
                    "x": int(box.xyxy[0][0]),
                    "y": int(box.xyxy[0][1]),
                    "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                    "height": int(box.xyxy[0][3] - box.xyxy[0][1]),
                    "label": model.names[int(box.cls)]
                })
        print("完整检测结果示例:", detections[:1])  # 显示第一个检测结果
        return {"results": detections}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Request error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")