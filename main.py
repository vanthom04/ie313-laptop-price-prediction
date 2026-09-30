import sys
from pathlib import Path

# Đảm bảo Python nhận diện đúng vị trí package trong thư mục src/
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.preprocessor import clean_and_preprocess


def run_pipeline():
    print("=" * 65)
    print(" 🚀 HỆ THỐNG PHÂN TÍCH VÀ MÔ HÌNH DỰ ĐOÁN GIÁ LAPTOP CŨ (IE313)")
    print("=" * 65)

    print("\n[1] ĐANG TẢI DỮ LIỆU THÔ...")
    raw_df = load_data("laptop_price.csv")

    if raw_df is not None:
        print("\n[2] ĐANG TIỀN XỬ LÝ DỮ LIỆU (NaN, One-Hot, Z-score)...")
        processed_df = clean_and_preprocess(raw_df)

        print("\n✅ KIỂM TRA DỮ LIỆU SAU XỬ LÝ (3 dòng đầu):")
        print(processed_df.iloc[:, :10].head(3))

        print("\n[HOÀN TẤT] Dữ liệu đã sẵn sàng cho phân tích EDA & Huấn luyện mô hình!")
    else:
        print("\n❌ Dừng pipeline vì không tìm thấy file dữ liệu.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:  # noqa: BLE001
        print(f"\nLỖI: {e}")