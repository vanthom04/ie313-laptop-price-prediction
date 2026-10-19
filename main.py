import sys
from pathlib import Path


# Đảm bảo Python nhận diện đúng vị trí package trong thư mục src/
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.preprocessor import clean_data, preprocess_data, remove_outliner
from src.eda import (
    select_categorical_variable, 
    select_numberical_variable, 
    get_num_cat_var, 
    numeric_var_statistic, 
    categorical_var_statistic, 
    plot_anova
)
from src.model_pipeline import train_data_pipeline


def run_pipeline():
    print("=" * 65)
    print(" 🚀 HỆ THỐNG PHÂN TÍCH VÀ MÔ HÌNH DỰ ĐOÁN GIÁ LAPTOP CŨ (IE313)")
    print("=" * 65)

    print("\n[1] ĐANG TẢI DỮ LIỆU THÔ...")
    raw_df = load_data("laptop_price.csv")

    if raw_df is not None:
        print("\n[HOÀN TẤT] Dữ liệu đã sẵn sàng cho phân tích EDA & Huấn luyện mô hình!")

        # 1. Làm sạch dữ liệu ban đầu
        df = clean_data(raw_df)
        df.head(3)

        # 2. Phân loại biến số & biến phân loại
        numerical_cols, categorical_cols = get_num_cat_var(df)

        # 3. Phân tích thống kê & Dashboard EDA
        print("\n[2] THỰC HIỆN THỐNG KÊ VÀ VẼ DASHBOARD EDA...")
        numeric_var_statistic(df, numerical_cols=numerical_cols)
        categorical_var_statistic(df, categorical_cols=categorical_cols)

        # 4. Xóa giá trị ngoại lệ (Outliers)
        print("\n[3] XỬ LÝ OUTLIERS...")
        df = remove_outliner(df, ["Price_euros"])
        df.head(3)

        # 5. Lọc đặc trưng (Feature Selection)
        print("\n[4] LỌC ĐẶC TRƯNG CÓ Ý NGHĨA THỐNG KÊ...")
        numerical = select_numberical_variable(
            df, series_name=numerical_cols, target='Price_euros', alpha=0.2
        )
        categorical = select_categorical_variable(df, categorical_cols)

        # Kiểm định ANOVA và vẽ biểu đồ p-value
        plot_anova(df, series=categorical_cols, target="Price_euros", alpha=0.05)

        # 6. Tiền xử lý dữ liệu chuẩn bị cho Model (Encoding, Scaling, Split)
        print("\n[5] TIỀN XỬ LÝ DỮ LIỆU BÀI TOÁN HỒI QUY...")
        procesed_df = preprocess_data(df, categorical, numerical, ignore_col=["Ips","Touchscreen"])

        # 7. Huấn luyện và đánh giá mô hình Pipeline
        print("\n[6] HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH PIPELINE...")
        r2_train, r2_test = train_data_pipeline(procesed_df, degree=1)

        print("\n" + "=" * 65)
        print(" 🎉 HOÀN THÀNH QUY TRÌNH PIPELINE TỰ ĐỘNG!")
        print(f" 📈 Kết quả R2 Train: {r2_train}")
        print(f" 📉 Kết quả R2 Test : {r2_test}")
        print("=" * 65)
    else:
        print("\n❌ Dừng pipeline vì không tìm thấy file dữ liệu.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:  # noqa: BLE001
        print(f"\nLỖI: {e}")