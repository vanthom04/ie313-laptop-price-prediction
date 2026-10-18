# Data Loader
import pandas as pd
import os


def load_data(file_name="laptop_price.csv"):
    """Hàm đọc dữ liệu Laptop từ thư mục data/raw/"""
    # Tìm đường dẫn tuyệt đối đến thư mục gốc của dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'raw', file_name)

    try:
        # Đọc file CSV (Dữ liệu Kaggle thường dùng encoding latin-1)
        df = pd.read_csv(file_path, encoding='latin-1')
        print(f"[THÀNH CÔNG] Đã tải dữ liệu. Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")
        return df
    except Exception as e:
        print(f"[LỖI] Không thể đọc file tại {file_path}. Chi tiết lỗi: {e}")
        return None