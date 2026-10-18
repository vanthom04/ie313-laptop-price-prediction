import sys
from pathlib import Path


# Đảm bảo Python nhận diện đúng vị trí package trong thư mục src/
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.preprocessor import clean_data, preprocess_data
from src.eda import select_categorical_variable, select_numberical_variable
from src.model_pipeline import train_data_pipeline


def run_pipeline():
    print("=" * 65)
    print(" 🚀 HỆ THỐNG PHÂN TÍCH VÀ MÔ HÌNH DỰ ĐOÁN GIÁ LAPTOP CŨ (IE313)")
    print("=" * 65)

    print("\n[1] ĐANG TẢI DỮ LIỆU THÔ...")
    raw_df = load_data("laptop_price.csv")

    if raw_df is not None:
        print("\n[HOÀN TẤT] Dữ liệu đã sẵn sàng cho phân tích EDA & Huấn luyện mô hình!")
        cleaned_df = clean_data(raw_df)

        numberic = select_numberical_variable(cleaned_df,["Weight","Ram","Inches","Touchscreen","Ips",
                "Cpu_Speed","SSD","HDD","Flash_Storage","Hybrid"], alpha = 0.2)
        
        categorical = select_categorical_variable(cleaned_df, ["Company","TypeName", "OpSys", 
                                       "Resolution", "Cpu_Brand", "Gpu_Type"])
        
        procesed_df = preprocess_data(cleaned_df, categorical, numberic, ignore_col=["Ips","Touchscreen"])

        print(train_data_pipeline(procesed_df))
    else:
        print("\n❌ Dừng pipeline vì không tìm thấy file dữ liệu.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:  # noqa: BLE001
        print(f"\nLỖI: {e}")