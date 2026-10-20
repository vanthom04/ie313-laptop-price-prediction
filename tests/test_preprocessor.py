"""tests/test_preprocessor.py
Kiểm thử các hàm làm sạch, xử lý ngoại lệ và tiền xử lý dữ liệu trong src/preprocessor.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.preprocessor import clean_data, preprocess_data, remove_outliner


def test_clean_data_none_raises_error():
    """Kiểm tra truyền None vào clean_data sẽ ném ngoại lệ ValueError."""
    with pytest.raises(ValueError, match="Dữ liệu đầu vào 'df' không được là None"):
        clean_data(None)


def test_clean_data_transformation(sample_raw_df: pd.DataFrame):
    """Kiểm tra tính đúng đắn của các cột trích xuất và kiểu dữ liệu sau clean_data."""
    cleaned = clean_data(sample_raw_df.copy(), target=TARGET_COL)

    # Các cột định danh và gốc phải bị loại bỏ
    dropped_cols = ["laptop_ID", "Product", "ScreenResolution", "Cpu", "Memory", "Gpu"]
    for col in dropped_cols:
        assert col not in cleaned.columns

    # Các cột trích xuất mới phải có mặt
    expected_new_cols = [
        "Ram",
        "Weight",
        "Touchscreen",
        "Ips",
        "Resolution",
        "Cpu_Speed",
        "Cpu_Brand",
        "SSD",
        "HDD",
        "Flash_Storage",
        "Hybrid",
        "Gpu_Brand",
        "Gpu_Type",
    ]
    for col in expected_new_cols:
        assert col in cleaned.columns

    # Kiểm tra kiểu dữ liệu chuẩn
    assert np.issubdtype(cleaned["Ram"].dtype, np.integer)
    assert np.issubdtype(cleaned["Weight"].dtype, np.floating)
    assert np.issubdtype(cleaned["Cpu_Speed"].dtype, np.floating)
    assert np.issubdtype(cleaned["Touchscreen"].dtype, np.integer)
    assert np.issubdtype(cleaned["Ips"].dtype, np.integer)


def test_clean_data_missing_values_imputation(sample_raw_df: pd.DataFrame):
    """Kiểm tra xử lý giá trị khuyết (missing values) bằng median và mode."""
    df_with_nan = sample_raw_df.copy()
    df_with_nan.loc[0, "Inches"] = np.nan
    df_with_nan.loc[1, "Weight"] = np.nan
    df_with_nan.loc[2, "Company"] = np.nan

    cleaned = clean_data(df_with_nan, target=TARGET_COL)
    assert cleaned["Inches"].isnull().sum() == 0
    assert cleaned["Weight"].isnull().sum() == 0
    assert cleaned["Company"].isnull().sum() == 0


def test_remove_outliner():
    """Kiểm tra hàm xóa ngoại lệ theo khoảng tứ phân vị IQR."""
    # Tạo phân phối bình thường với 1 điểm ngoại lệ cực lớn
    df = pd.DataFrame({"Price_euros": [100.0, 110.0, 105.0, 95.0, 102.0, 108.0, 5000.0]})
    filtered_df = remove_outliner(df, cols=["Price_euros"])

    assert len(filtered_df) < len(df)
    assert 5000.0 not in filtered_df["Price_euros"].values


def test_remove_outliner_missing_column():
    """Kiểm tra khi truyền cột không tồn tại thì không làm thay đổi dữ liệu."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    filtered_df = remove_outliner(df, cols=["NonExistentColumn"])
    assert len(filtered_df) == 3


def test_preprocess_data(sample_clean_df: pd.DataFrame):
    """Kiểm tra One-Hot Encoding và chuẩn hóa Z-score."""
    categorical_cols = ["Company", "TypeName"]
    numeric_cols = ["Inches", "Ram", "Weight"]
    ignore_cols = ["Touchscreen", "Ips"]

    processed = preprocess_data(
        sample_clean_df.copy(),
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        ignore_col=ignore_cols,
        target=TARGET_COL,
    )

    # Biến mục tiêu phải được giữ nguyên
    assert TARGET_COL in processed.columns

    # Các cột trong ignore_cols hoặc cột ngoài danh sách phải bị loại
    for col in ignore_cols:
        assert col not in processed.columns

    # Biến số sau Z-score scaler phải có trung bình xấp xỉ 0 (sai số dung sai)
    for col in numeric_cols:
        mean_val = processed[col].mean()
        assert abs(mean_val) < 1e-6
