import sys
from pathlib import Path

# Đảm bảo Python nhận diện đúng vị trí package trong thư mục src/
sys.path.append(str(Path(__file__).resolve().parent))

# from src import config


def run_pipeline():
    print("=" * 65)
    print(" 🚀 HỆ THỐNG PHÂN TÍCH VÀ MÔ HÌNH DỰ ĐOÁN GIÁ LAPTOP CỦ (IE313)")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:  # noqa: BLE001
        print(f"\nLỖI: {e}")
