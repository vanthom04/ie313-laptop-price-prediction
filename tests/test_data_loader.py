"""tests/test_data_loader.py
Kiểm thử chức năng nạp dữ liệu từ file CSV trong src/data_loader.py.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH
from src.data_loader import load_data


def test_load_data_default_path():
    """Kiểm tra tải dữ liệu từ đường dẫn mặc định RAW_DATA_PATH."""
    df = load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 1303
    assert df.shape[1] == 13


def test_load_data_string_filename():
    """Kiểm tra tải dữ liệu bằng tên file chuỗi trong data/raw/."""
    df = load_data("laptop_price.csv")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_custom_path_object():
    """Kiểm tra tải dữ liệu bằng Path object."""
    df = load_data(Path(RAW_DATA_PATH))
    assert df is not None
    assert isinstance(df, pd.DataFrame)


def test_load_data_non_existent_file():
    """Kiểm tra xử lý khi file không tồn tại trả về None thay vì crash."""
    df = load_data("file_khong_ton_tai_12345.csv")
    assert df is None


def test_load_data_columns_integrity():
    """Kiểm tra các cột bắt buộc phải có trong dữ liệu thô."""
    expected_cols = {
        "laptop_ID",
        "Company",
        "Product",
        "TypeName",
        "Inches",
        "ScreenResolution",
        "Cpu",
        "Ram",
        "Memory",
        "Gpu",
        "OpSys",
        "Weight",
        "Price_euros",
    }
    df = load_data()
    assert df is not None
    assert expected_cols.issubset(set(df.columns))
