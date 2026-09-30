import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_and_preprocess(df):
    """Hàm làm sạch, Xử lý NaN, One-Hot Encoding và chuẩn hóa Z-score"""
    print("Đang tiến hành tiền xử lý dữ liệu...")

    # 1. Xóa các cột không mang ý nghĩa dự báo
    cols_to_drop = ['laptop_ID', 'Product']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    df = df.dropna().copy()

    if 'Ram' in df.columns:
        # Ép về chuỗi -> Xóa chữ 'GB' -> Xóa khoảng trắng thừa -> Ép về số nguyên
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).str.strip().astype(int)

    if 'Weight' in df.columns:
        # Ép về chuỗi -> Xóa 'kg' (và 'kgs' nếu có) -> Ép về số thực
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).str.replace('s', '',
                                                                                               regex=False).str.strip().astype(
            float)

    # Định nghĩa các biến phân loại và biến số
    categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    numeric_cols = ['Inches', 'Ram', 'Weight']

    cat_cols_exist = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols_exist, drop_first=True)

    num_cols_exist = [c for c in numeric_cols if c in df.columns]
    if num_cols_exist:
        scaler = StandardScaler()
        df_encoded[num_cols_exist] = scaler.fit_transform(df_encoded[num_cols_exist])

    print(
        f"[THÀNH CÔNG] Tiền xử lý hoàn tất! Kích thước dữ liệu mới: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột.")
    return df_encoded