from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, RAW_DATA_PATH


def load_data(file_name: str | Path = RAW_DATA_PATH) -> pd.DataFrame | None:
    """Hàm đọc dữ liệu Laptop từ thư mục data/raw/ hoặc đường dẫn tùy chọn."""
    if isinstance(file_name, Path):
        file_path = file_name
    elif Path(file_name).is_file():
        file_path = Path(file_name)
    else:
        file_path = DATA_DIR / "raw" / file_name

    try:
        # Đọc file CSV (Dữ liệu Kaggle thường dùng encoding latin-1)
        df = pd.read_csv(file_path, encoding="latin-1")
        print(f"[THÀNH CÔNG] Đã tải dữ liệu. Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")
        return df
    except Exception as e:
        print(f"[LỖI] Không thể đọc file tại {file_path}. Chi tiết lỗi: {e}")
        return None
