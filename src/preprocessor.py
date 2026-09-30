import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_and_preprocess(df):
    """Hàm làm sạch, Điền khuyết (Median/Mode), One-Hot Encoding và chuẩn hóa Z-score"""
    print("Đang tiến hành tiền xử lý dữ liệu (Áp dụng Median/Mode cho NaN)...")

    # 1. Xóa các cột không mang ý nghĩa thống kê
    cols_to_drop = ['laptop_ID', 'Product']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 2. Làm sạch cột 'Ram' và 'Weight' để chuyển thành số
    # Dùng float tạm thời để không bị lỗi nếu ô đó đang là NaN
    if df['Ram'].dtype == object:
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(float)
    if df['Weight'].dtype == object:
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    # Phân loại các nhóm biến
    categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    numeric_cols = ['Inches', 'Ram', 'Weight']

    # 3. XỬ LÝ GIÁ TRỊ KHUYẾT (Missing Values) theo đúng mô tả README
    # Điền giá trị Trung vị (Median) cho các cột số
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Điền giá trị Xuất hiện nhiều nhất (Mode) cho các cột phân loại
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ép kiểu Ram về lại số nguyên (int) sau khi đã điền khuyết xong
    df['Ram'] = df['Ram'].astype(int)

    # 4. One-Hot Encoding cho các biến phân loại
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 5. Chuẩn hóa Z-score cho biến số (đưa về phân phối chuẩn)
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    print(
        f"[THÀNH CÔNG] Tiền xử lý hoàn tất! Kích thước dữ liệu mới: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột.")
    return df_encoded