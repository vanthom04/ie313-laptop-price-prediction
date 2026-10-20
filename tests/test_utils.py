"""tests/test_utils.py
Kiểm thử các hàm tiện ích trích xuất và biến đổi chuỗi trong src/utils.py.
"""

import pandas as pd
import pytest

from src.utils import extract_cap, extract_cpu_brand, process_gpu_type


@pytest.mark.parametrize(
    ("cpu_input", "expected_brand"),
    [
        ("Intel Core i7 7500U 2.7GHz", "Intel Core i7"),
        ("Intel Core i5 7200U 2.5GHz", "Intel Core i5"),
        ("Intel Core i3 6006U 2GHz", "Intel Core i3"),
        ("Intel Celeron Dual Core N3060 1.6GHz", "Other Intel Processor"),
        ("Intel Pentium Quad Core N4200 1.1GHz", "Other Intel Processor"),
        ("Intel Xeon E3-1535M v6 3.1GHz", "Other Intel Processor"),
        ("AMD A9-Series 9420 3GHz", "AMD Processor"),
        ("AMD Ryzen 7 1700 3.0GHz", "AMD Processor"),
        ("Samsung Cortex A72&A53 2.0GHz", "Other"),
    ],
)
def test_extract_cpu_brand(cpu_input: str, expected_brand: str):
    """Kiểm tra độ chính xác phân loại CPU brand."""
    assert extract_cpu_brand(cpu_input) == expected_brand


def test_extract_cap():
    """Kiểm tra trích xuất dung lượng ổ cứng từ series chuỗi."""
    mem_series = pd.Series(["128 SSD", "512 SSD", "1000 HDD", "256 Flash Storage", ""])

    ssd_cap = extract_cap(mem_series, "SSD")
    assert ssd_cap.tolist() == [128, 512, 0, 0, 0]

    hdd_cap = extract_cap(mem_series, "HDD")
    assert hdd_cap.tolist() == [0, 0, 1000, 0, 0]

    flash_cap = extract_cap(mem_series, "Flash Storage")
    assert flash_cap.tolist() == [0, 0, 0, 256, 0]

    hybrid_cap = extract_cap(mem_series, "Hybrid")
    assert hybrid_cap.tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("gpu_input", "expected_type"),
    [
        ("Nvidia Quadro M1200", "Nvidia Quadro"),
        ("Nvidia GeForce GTX 1050", "Nvidia GeForce"),
        ("AMD FirePro W4190M", "AMD FirePro"),
        ("AMD Radeon RX 580", "AMD Radeon"),
        ("Intel HD Graphics 620", "Intel Integrated Graphics"),
        ("Intel Iris Plus Graphics 640", "Intel Integrated Graphics"),
        ("ARM Mali T860 MP4", "Other"),
    ],
)
def test_process_gpu_type(gpu_input: str, expected_type: str):
    """Kiểm tra độ chính xác phân loại loại GPU."""
    assert process_gpu_type(gpu_input) == expected_type
