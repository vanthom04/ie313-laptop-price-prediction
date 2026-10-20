"""tests/test_config.py
Kiểm thử các hằng số và đường dẫn cấu hình trong src/config.py.
"""

from pathlib import Path

from src.config import (
    ALPHA_ANOVA,
    ALPHA_CORR,
    BASE_DIR,
    DATA_DIR,
    POLY_DEGREE,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    REPORTS_IMG_DIR,
    TARGET_COL,
    TEST_SIZE,
)


def test_paths_configuration():
    """Kiểm tra tính hợp lệ của các đường dẫn hệ thống."""
    assert isinstance(BASE_DIR, Path)
    assert BASE_DIR.is_dir()

    assert isinstance(DATA_DIR, Path)
    assert DATA_DIR.is_dir()

    assert isinstance(RAW_DATA_PATH, Path)
    assert RAW_DATA_PATH.suffix == ".csv"

    assert isinstance(PROCESSED_DATA_PATH, Path)
    assert PROCESSED_DATA_PATH.suffix == ".csv"

    assert isinstance(REPORTS_DIR, Path)
    assert isinstance(REPORTS_IMG_DIR, Path)


def test_model_hyperparameters():
    """Kiểm tra các tham số mô hình và ngưỡng thống kê."""
    assert TARGET_COL == "Price_euros"
    assert isinstance(RANDOM_STATE, int)
    assert 0.0 < TEST_SIZE < 1.0
    assert POLY_DEGREE >= 1
    assert 0.0 <= ALPHA_CORR <= 1.0
    assert 0.0 < ALPHA_ANOVA < 1.0
