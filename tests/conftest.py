"""tests/conftest.py
Cấu hình pytest fixtures dùng chung và thiết lập môi trường headless cho matplotlib.
"""

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TARGET_COL
from src.preprocessor import clean_data, preprocess_data


@pytest.fixture(autouse=True)
def disable_matplotlib_show(monkeypatch: pytest.MonkeyPatch):
    """Vô hiệu hóa plt.show() trong tất cả các test để tránh bật giao diện và treo tiến trình."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture tạo DataFrame thô cho laptop_price với mẫu đủ lớn cho kiểm định thống kê."""
    data = {
        "laptop_ID": list(range(1, 11)),
        "Company": [
            "Apple",
            "Dell",
            "HP",
            "Apple",
            "Dell",
            "HP",
            "Lenovo",
            "Lenovo",
            "Asus",
            "Asus",
        ],
        "Product": [
            "MacBook Pro",
            "XPS 13",
            "Pavilion",
            "MacBook Air",
            "Inspiron",
            "Envy",
            "ThinkPad",
            "IdeaPad",
            "ZenBook",
            "ROG",
        ],
        "TypeName": [
            "Ultrabook",
            "Ultrabook",
            "Notebook",
            "Ultrabook",
            "Notebook",
            "Notebook",
            "Notebook",
            "Notebook",
            "Gaming",
            "Gaming",
        ],
        "Inches": [13.3, 13.3, 15.6, 13.3, 15.6, 15.6, 14.0, 15.6, 15.6, 17.3],
        "ScreenResolution": [
            "IPS Panel Retina Display 2560x1600",
            "Full HD / Touchscreen 1920x1080",
            "1920x1080",
            "IPS Panel Retina Display 2560x1600",
            "1920x1080",
            "IPS Panel Full HD 1920x1080",
            "IPS Panel Full HD 1920x1080",
            "1366x768",
            "Full HD 1920x1080",
            "Full HD 1920x1080",
        ],
        "Cpu": [
            "Intel Core i5 2.3GHz",
            "Intel Core i7 7500U 2.7GHz",
            "Intel Core i3 7100U 2.4GHz",
            "Intel Core i5 1.8GHz",
            "Intel Core i5 7200U 2.5GHz",
            "Intel Core i7 8550U 1.8GHz",
            "Intel Celeron Dual Core N3350 1.1GHz",
            "Intel Core i3 6006U 2.0GHz",
            "AMD Ryzen 7 1700 3.0GHz",
            "Intel Core i7 7700HQ 2.8GHz",
        ],
        "Ram": ["8GB", "16GB", "8GB", "8GB", "8GB", "16GB", "4GB", "4GB", "16GB", "32GB"],
        "Memory": [
            "128GB SSD",
            "512GB SSD",
            "1TB HDD",
            "256GB SSD",
            "256GB SSD + 1TB HDD",
            "1TB Hybrid",
            "128GB SSD + 1TB HDD",
            "500GB HDD",
            "256GB Flash Storage",
            "512GB SSD + 1TB HDD",
        ],
        "Gpu": [
            "Intel Iris Plus Graphics 640",
            "Intel HD Graphics 620",
            "Nvidia GeForce GTX 1050",
            "Intel HD Graphics 6000",
            "Intel HD Graphics 620",
            "Nvidia GeForce 940MX",
            "Intel HD Graphics 500",
            "Intel HD Graphics 520",
            "AMD Radeon RX 580",
            "Nvidia GeForce GTX 1070",
        ],
        "OpSys": [
            "macOS",
            "Windows 10",
            "Windows 10",
            "macOS",
            "Windows 10",
            "Windows 10",
            "Windows 10",
            "Windows 10",
            "Windows 10",
            "Windows 10",
        ],
        "Weight": [
            "1.37kg",
            "1.2kg",
            "1.86kg",
            "1.34kg",
            "2.1kg",
            "1.9kg",
            "1.4kg",
            "2.2kg",
            "2.2kg",
            "3.0kg",
        ],
        "Price_euros": [
            1339.69,
            1499.00,
            650.00,
            999.00,
            750.00,
            1199.00,
            399.00,
            350.00,
            1799.00,
            2499.00,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_clean_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture trả về DataFrame đã được làm sạch qua clean_data."""
    return clean_data(sample_raw_df.copy(), target=TARGET_COL)


@pytest.fixture
def sample_preprocessed_df(sample_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture trả về DataFrame đã mã hóa và chuẩn hóa qua preprocess_data."""
    categorical_cols = ["Company", "TypeName", "OpSys", "Cpu_Brand", "Gpu_Brand", "Gpu_Type"]
    numeric_cols = ["Inches", "Ram", "Weight", "Cpu_Speed", "SSD", "HDD"]
    return preprocess_data(
        sample_clean_df.copy(),
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        ignore_col=["Ips", "Touchscreen", "Flash_Storage", "Hybrid", "Resolution"],
        target=TARGET_COL,
    )
