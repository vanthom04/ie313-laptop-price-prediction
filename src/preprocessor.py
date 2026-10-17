import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import extract_cpu_brand, extract_cap, process_gpu_type

def clean_data(df: pd.DataFrame, target = 'Price_euros') -> pd.DataFrame:
    """Hàm làm sạch dữ liệu thô, xử lý giá trị khuyết và trích xuất các đặc trưng."""
    print("Đang tiến hành làm sạch dữ liệu...")
    df = df.copy()

    # 1. Xóa các cột định danh không mang ý nghĩa thống kê
    cols_to_drop = ['laptop_ID', 'Product']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore', inplace=True)

    # 2. Xóa ngoại lệ (Outliers) trên biến mục tiêu Price_euros
    if target in df.columns:
        q1 = df[target].quantile(0.25)
        q3 = df[target].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        df = df[df[target] <= upper_bound]

    # 3. Chuẩn hóa dạng dữ liệu Ram và Weight
    df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).astype(float)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).astype(float)

    # 4. Xử lý giá trị khuyết (Missing Values)
    categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
    numeric_cols = ['Inches', 'Ram', 'Weight']

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    df['Ram'] = df['Ram'].astype(int)

    # 5. Trích xuất đặc trưng từ ScreenResolution
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
    df['Ips'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)
    df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
    df.drop(columns=['ScreenResolution'], inplace=True)

    # 6. Trích xuất đặc trưng từ Cpu
    df['Cpu_Speed'] = df['Cpu'].str.extract(r'(\d+\.?\d*)\s*GHz', expand=False).astype(float)
    df['Cpu_Brand'] = df['Cpu'].apply(extract_cpu_brand)
    df.drop(columns=['Cpu'], inplace=True)

    # 7. Trích xuất đặc trưng từ Memory (Dung lượng ổ cứng)
    memory_clean = (
        df['Memory']
        .astype(str)
        .str.replace('.0', '', regex=False)
        .str.replace('GB', '', regex=False)
        .str.replace('TB', '000', regex=False)
    )

    split_mem = memory_clean.str.split('+', expand=True)
    mem1 = split_mem[0].str.strip()
    mem2 = split_mem[1].str.strip() if 1 in split_mem.columns else pd.Series('', index=df.index)

    df['SSD'] = extract_cap(mem1, 'SSD') + extract_cap(mem2, 'SSD')
    df['HDD'] = extract_cap(mem1, 'HDD') + extract_cap(mem2, 'HDD')
    df['Flash_Storage'] = extract_cap(mem1, 'Flash Storage') + extract_cap(mem2, 'Flash Storage')
    df['Hybrid'] = extract_cap(mem1, 'Hybrid') + extract_cap(mem2, 'Hybrid')
    df.drop(columns=['Memory'], inplace=True)

    # 8. Trích xuất đặc trưng từ Gpu
    df['Gpu_Brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])
    df['Gpu_Type'] = df['Gpu'].apply(process_gpu_type)
    df.drop(columns=['Gpu'], inplace=True)

    print(f"[THÀNH CÔNG] Làm sạch dữ liệu hoàn tất! Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df


def preprocess_data(df, categorical_cols : list, numeric_cols  : list, ignore_col  : list, target = "Price_euros"):
    """Hàm One-Hot Encoding và chuẩn hóa Z-score"""
    print("Đang tiến hành tiền xử lý dữ liệu ...")
    all_column = df.columns.to_list()
    
    droped_features = [
        col
        for col in all_column
        if col not in categorical_cols
        and col not in numeric_cols
        and col != target
    ]
    
    # 1. Phân loại các nhóm biến
    categorical_cols = [col for col in categorical_cols if col not in ignore_col]
    numeric_cols = [col for col in numeric_cols if col not in ignore_col]

    # 2. Xóa các biến không cần thiết
    df.drop(columns=droped_features, inplace=True)
    
    # 3. One-Hot Encoding cho các biến phân loại
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 4. Chuẩn hóa Z-score cho biến số (đưa về phân phối chuẩn)
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    print(
        f"[THÀNH CÔNG] Tiền xử lý hoàn tất! Kích thước dữ liệu mới: {df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột.")
    return df_encoded

