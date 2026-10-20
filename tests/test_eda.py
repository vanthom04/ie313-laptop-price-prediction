"""tests/test_eda.py
Kiểm thử các hàm thống kê, phân tích tương quan, ANOVA và trực quan hóa trong src/eda.py.
"""

import pandas as pd
from scipy.stats._stats_py import F_onewayResult

from src.config import TARGET_COL
from src.eda import (
    anova_analisis,
    box_plot,
    categorical_var_statistic,
    correlation,
    get_num_cat_var,
    hist_plot,
    numeric_var_statistic,
    plot_anova,
    plot_single_boxplot,
    select_categorical_variable,
    select_numberical_variable,
)


def test_get_num_cat_var(sample_clean_df: pd.DataFrame):
    """Kiểm tra phân loại chính xác các cột số và cột phân loại."""
    num_cols, cat_cols = get_num_cat_var(sample_clean_df)

    assert "Price_euros" in num_cols
    assert "Ram" in num_cols
    assert "Weight" in num_cols

    assert "Company" in cat_cols
    assert "TypeName" in cat_cols
    assert "OpSys" in cat_cols

    # Không có cột nào bị trùng giữa 2 nhóm
    assert set(num_cols).isdisjoint(set(cat_cols))


def test_correlation(sample_clean_df: pd.DataFrame):
    """Kiểm tra tính ma trận tương quan giữa các biến số."""
    num_cols, _ = get_num_cat_var(sample_clean_df)
    corr_matrix = correlation(sample_clean_df, num_cols)

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (len(num_cols), len(num_cols))
    # Đường chéo chính của các biến có phương sai khác 0 luôn bằng 1.0
    for col in num_cols:
        if sample_clean_df[col].std() > 0:
            assert round(corr_matrix.loc[col, col], 2) == 1.0


def test_select_numberical_variable(sample_clean_df: pd.DataFrame):
    """Kiểm tra lọc biến số có độ tương quan vượt ngưỡng alpha."""
    num_cols, _ = get_num_cat_var(sample_clean_df)
    selected = select_numberical_variable(
        sample_clean_df, series_name=num_cols, target=TARGET_COL, alpha=0.2
    )

    assert isinstance(selected, list)
    # Biến mục tiêu không được nằm trong danh sách các biến độc lập được chọn
    assert TARGET_COL not in selected


def test_anova_analisis(sample_clean_df: pd.DataFrame):
    """Kiểm tra kiểm định ANOVA giữa một biến phân loại và biến mục tiêu."""
    res = anova_analisis(sample_clean_df, "Company", target=TARGET_COL)
    assert isinstance(res, F_onewayResult)
    assert hasattr(res, "statistic")
    assert hasattr(res, "pvalue")
    assert res.pvalue >= 0.0


def test_select_categorical_variable(sample_clean_df: pd.DataFrame):
    """Kiểm tra lọc các biến phân loại có p-value < alpha."""
    _, cat_cols = get_num_cat_var(sample_clean_df)
    selected = select_categorical_variable(
        sample_clean_df, series=cat_cols, target=TARGET_COL, alpha=0.99
    )
    assert isinstance(selected, list)


def test_eda_visualizations_execute_cleanly(sample_clean_df: pd.DataFrame):
    """Kiểm tra các hàm vẽ biểu đồ thực thi trơn tru mà không vấp lỗi."""
    num_cols, cat_cols = get_num_cat_var(sample_clean_df)

    # 1. Box plot
    box_plot(x_axis="TypeName", y_axis=TARGET_COL, data=sample_clean_df)
    plot_single_boxplot(sample_clean_df, column=TARGET_COL)

    # 2. Hist plot
    hist_plot(x_axis="Ram", data=sample_clean_df)

    # 3. ANOVA plot
    plot_anova(sample_clean_df, series=cat_cols[:2], target=TARGET_COL)

    # 4. Dashboard statistics
    numeric_var_statistic(sample_clean_df, numerical_cols=num_cols[:3])
    categorical_var_statistic(sample_clean_df, categorical_cols=cat_cols[:2])
