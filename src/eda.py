# EDA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

def box_plot(**kwargs):
    sns.boxplot(x=kwargs["x_axis"], y=kwargs["y_axis"], data=kwargs["data"])
    plt.xticks(rotation=15)
    plt.show()

def plot_single_boxplot(data, column, figsize=(6, 8), title=None):
    """Vẽ biểu đồ Boxplot đơn cho một biến số (tương tự df[column].plot(kind='box'))

    Tham số:
        data: DataFrame
        column: Tên cột biến số cần vẽ (vd: 'Price_euros')
        figsize: Kích thước biểu đồ
        title: Tiêu đề biểu đồ
    """
    plt.figure(figsize=figsize)

    # Dùng seaborn với y=column để vẽ Boxplot dọc
    sns.boxplot(y=data[column])

    if title:
        plt.title(title)
    else:
        plt.title(f"Boxplot của {column}")

    plt.ylabel(column)
    plt.show()

def hist_plot(**kwargs):
    sns.histplot(x=kwargs["x_axis"], data=kwargs["data"], kde=True)
    plt.xticks(rotation=15)
    plt.show()

def correlation(data, series_name: list):
    df_corr = data[series_name].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm",
    xticklabels=df_corr.columns, yticklabels=df_corr.index, linewidths=0.5)
    plt.title("Matrix Heatmap (Seaborn)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()
    return df_corr

def select_numberical_variable(data, series_name: list, target = "Price_euros", alpha = 0.3):
    #tinh correlation cho tung bien, voi corr (<-0.3 or > 0.3)
    series_name.append(target)
    corr_matrix = correlation(data, series_name)
    abs_corr = corr_matrix[target].abs()
    selected_features_abs = abs_corr[
        (abs_corr > alpha) & (abs_corr.index != "Price_euros")
    ].index.tolist()

    print(f"Các cột có |corr| > {alpha} với {target}:")
    print(selected_features_abs)
    return selected_features_abs

def anova_analisis(data, series, target):
    df_anova = data[[series, target]]

    # 1. Tự động lấy tất cả các nhóm dữ liệu có trong cột TypeName
    groups = [
        group[target].values for _, group in df_anova.groupby(series)
    ]

    # 2. Chạy kiểm định One-way ANOVA cho tất cả các nhóm cùng lúc
    anova_result = stats.f_oneway(*groups)

    return anova_result

def select_categorical_variable(data, series: list, target = "Price_euros", alpha = 0.05):
    #tinh anova voi p-value < 0.05
    selected_features = []
    for s in series:
            anova_result = anova_analisis(data, s, target)
            if anova_result.pvalue < alpha:
                selected_features.append(s)
                print(f"Biến {s} đã được chọn làm biến đặc trưng")
                print(f"\tF-test của {s}: {anova_result.statistic:.4f}")
                print(f"\tp-value của {s}:: {anova_result.pvalue:.4e}")
    return selected_features

