"""src/config.py
Mô-đun quản lý đường dẫn và hằng số cấu hình cho toàn bộ dự án.
"""

import io
import sys
from pathlib import Path

# Đảm bảo console Windows in ký tự UTF-8 không bị lỗi charmap
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

# Đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn các thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "laptop_price.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_laptop_price.csv"

# Đường dẫn lưu báo cáo & hình ảnh xuất ra
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_IMG_DIR = REPORTS_DIR / "img"

# Cấu hình hằng số cho Bài toán & Mô hình
TARGET_COL = "Price_euros"  # Biến mục tiêu cần dự báo
RANDOM_STATE = 42  # Cố định seed để kết quả đồng nhất giữa các lần chạy
TEST_SIZE = 0.2  # Tỷ lệ chia tập Test (20%)
POLY_DEGREE = 1  # Bậc đa thức cho PolynomialFeatures trong Pipeline
ALPHA_CORR = 0.2  # Ngưỡng tương quan Pearson tối thiểu để lọc biến số
ALPHA_ANOVA = 0.05  # Mức ý nghĩa thống kê p-value để lọc biến phân loại qua ANOVA
