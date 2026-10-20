"""tests/test_integration.py
Kiểm thử tích hợp đầu-cuối (End-to-End Integration) cho toàn bộ quy trình dự đoán giá laptop.
"""

from main import run_pipeline
from src.config import (
    ALPHA_ANOVA,
    ALPHA_CORR,
    POLY_DEGREE,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.data_loader import load_data
from src.eda import (
    get_num_cat_var,
    select_categorical_variable,
    select_numberical_variable,
)
from src.model_pipeline import train_data_pipeline
from src.preprocessor import clean_data, preprocess_data, remove_outliner


def test_end_to_end_data_to_model_pipeline():
    """Kiểm thử toàn bộ chuỗi xử lý từ file dữ liệu thô đến kết quả huấn luyện mô hình."""
    # Đọc dữ liệu
    df_raw = load_data(RAW_DATA_PATH)
    assert df_raw is not None
    assert len(df_raw) > 1000

    # Làm sạch
    df_clean = clean_data(df_raw, target=TARGET_COL)
    assert df_clean.shape[0] == df_raw.shape[0]

    # Phân loại biến
    numerical_cols, categorical_cols = get_num_cat_var(df_clean)
    assert len(numerical_cols) > 0
    assert len(categorical_cols) > 0

    # Lọc ngoại lệ
    df_no_outlier = remove_outliner(df_clean, [TARGET_COL])
    assert len(df_no_outlier) <= len(df_clean)

    # Lọc biến có ý nghĩa
    num_selected = select_numberical_variable(
        df_no_outlier, series_name=numerical_cols, target=TARGET_COL, alpha=ALPHA_CORR
    )
    cat_selected = select_categorical_variable(
        df_no_outlier, series=categorical_cols, target=TARGET_COL, alpha=ALPHA_ANOVA
    )
    assert len(num_selected) > 0
    assert len(cat_selected) > 0

    # Tiền xử lý (One-hot + Scaling)
    df_preprocessed = preprocess_data(
        df_no_outlier,
        categorical_cols=cat_selected,
        numeric_cols=num_selected,
        ignore_col=["Ips", "Touchscreen"],
        target=TARGET_COL,
    )
    assert TARGET_COL in df_preprocessed.columns

    # Huấn luyện mô hình
    r2_train, r2_test = train_data_pipeline(df_preprocessed, degree=POLY_DEGREE, target=TARGET_COL)

    # Khẳng định chất lượng mô hình đạt ngưỡng chấp nhận được
    assert r2_train > 0.75
    assert r2_test > 0.75


def test_main_run_pipeline_execution():
    """Kiểm thử hàm run_pipeline() trong main.py thực thi thành công từ đầu tới cuối."""
    # Đã có disable_matplotlib_show trong conftest.py nên run_pipeline sẽ không bị treo
    run_pipeline()
