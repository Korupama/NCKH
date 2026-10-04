"""Uploaded source videos and indexed frame decoding for the replay editor."""
from pathlib import Path
import math
import uuid
import shutil
import cv2
import base64
import numpy as np


def analysis_source_frame(directory, request):
    """Explicit displayed/preview pixels take priority over video re-decoding."""
    kind = request.get('source_kind')
    if kind in {'DISPLAYED_VIDEO_FRAME', 'INDEXED_FRAME_PREVIEW'} or ('image' in request and 'video_id' not in request):
        image = request.get('image', '')
        if not isinstance(image, str) or ',' not in image:
            raise ValueError('Thiếu ảnh frame đã chọn. Vui lòng chọn lại frame.')
        header, data = image.split(',', 1)
        if header not in {'data:image/png;base64', 'data:image/jpeg;base64'}:
            raise ValueError('Ảnh nguồn phải là PNG hoặc JPEG.')
        try:
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, cv2.error) as exc:
            raise ValueError('Không giải mã được ảnh frame đã chọn.') from exc
        if pixels is None:
            raise ValueError('Không giải mã được ảnh frame đã chọn.')
        return pixels, kind or 'SUBMITTED_IMAGE'
    if kind is not None:
        raise ValueError('Nguồn frame không hợp lệ.')
    if 'video_id' in request:
        return read_video_frame(directory, request['video_id'], request.get('estimated_frame', 0)), 'INDEXED_VIDEO_DECODE'
    raise ValueError('Cần ảnh frame nguồn để phân tích.')


def validate_upload(size, directory):
    if size <= 0:
        raise ValueError("Video rỗng hoặc thiếu thông tin dung lượng.")
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if size + 256 * 1024 ** 2 > free:
        raise ValueError("Không đủ dung lượng lưu video trên máy chủ: cần %.2f GB, còn %.2f GB." %
                         (size / 1024 ** 3, free / 1024 ** 3))


def register_video(stream, size, directory):
    validate_upload(size, directory)
    token = uuid.uuid4().hex
    path = directory / (token + ".video")
    try:
        remaining = size
        with path.open("wb") as target:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Tải video chưa hoàn tất.")
                target.write(chunk)
                remaining -= len(chunk)
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ok, _ = cap.read()
            if not ok or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("Máy chủ không giải mã được video hoặc thông tin frame. Hãy dùng MP4 mã hóa H.264; video phát được trong trình duyệt vẫn có thể thiếu codec ở máy chủ.")
        finally:
            cap.release()
        return {"video_id": token, "fps": fps, "frame_count": int(count)}
    except Exception:
        path.unlink(missing_ok=True)
        raise


def read_video_frame(directory, token, index):
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Video không hợp lệ. Vui lòng chọn lại video.")
    if isinstance(index, bool) or int(index) != float(index):
        raise ValueError("Số frame phải là số nguyên.")
    index = int(index)
    cap = cv2.VideoCapture(str(Path(directory) / (token + ".video")))
    try:
        if not cap.isOpened() or not 0 <= index < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError("Frame nằm ngoài video hoặc video không còn tồn tại.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Không giải mã được frame đã chọn.")
        return frame
    finally:
        cap.release()
