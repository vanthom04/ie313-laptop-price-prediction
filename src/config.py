"""
src/config.py
Mô-đun quản lý đường dẫn và hằng số cấu hình cho toàn bộ dự án.
"""

from pathlib import Path

# Đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn các thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "laptop_price.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_laptop_price.csv"

# Đường dẫn lưu báo cáo & hình ảnh xuất ra
REPORTS_DIR = BASE_DIR / "reports"

# Cấu hình hằng số cho Bài toán & Mô hình
TARGET_COL = "Price"  # Biến mục tiêu cần dự báo (hoặc price)
RANDOM_STATE = 42  # Cố định seed để kết quả đồng nhất giữa các lần chạy
TEST_SIZE = 0.2  # Tỷ lệ chia tập Test (20%)
POLY_DEGREE = 2  # Bậc đa thức cho PolynomialFeatures trong Pipeline
