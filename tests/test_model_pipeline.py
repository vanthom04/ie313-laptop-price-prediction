"""tests/test_model_pipeline.py
Kiểm thử quy trình huấn luyện và đánh giá mô hình trong src/model_pipeline.py.
"""

import numpy as np
import pandas as pd

from src.config import TARGET_COL
from src.model_pipeline import train_data_pipeline


def test_train_data_pipeline_synthetic_dataset():
    """Kiểm tra huấn luyện pipeline trên tập dữ liệu tổng hợp."""
    np.random.seed(42)
    n_samples = 40

    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    # y có quan hệ tuyến tính mạnh với x1 và x2
    y = 500 + 200 * x1 - 100 * x2 + np.random.normal(0, 10, n_samples)

    df = pd.DataFrame({"Feature1": x1, "Feature2": x2, TARGET_COL: y})

    train_score, test_score = train_data_pipeline(
        df, degree=1, target=TARGET_COL, test_size=0.25, random_state=42
    )

    assert isinstance(train_score, float)
    assert isinstance(test_score, float)
    # Với dữ liệu tuyến tính rõ ràng, R2 train và test phải > 0.8
    assert train_score > 0.8
    assert test_score > 0.8


def test_train_data_pipeline_degree_parameter():
    """Kiểm tra tham số degree đa thức trong pipeline."""
    np.random.seed(42)
    n_samples = 50
    x = np.random.uniform(-3, 3, n_samples)
    y = 2 * (x**2) + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame({"X": x, TARGET_COL: y})

    # Bậc 2 phù hợp cho hàm phi tuyến bậc 2
    train_score_deg2, _ = train_data_pipeline(
        df, degree=2, target=TARGET_COL, test_size=0.2, random_state=42
    )
    assert train_score_deg2 > 0.8
