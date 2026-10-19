# EDA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import math

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

def select_numberical_variable(
    data, series_name, target='Price_euros', alpha=0.2
):
  corr_matrix = correlation(data, series_name)
  abs_corr = corr_matrix[target].abs()

  # Lọc điều kiện đúng cú pháp Pandas Series
  selected_features_abs = abs_corr[
      (abs_corr > alpha) & (abs_corr.index.to_series() != target)
  ].index.tolist()

  print(f'Các cột có |corr| > {alpha} với {target}:')
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
    return selected_features

def plot_anova(
    data, series: list, target="Price_euros", alpha=0.05, figsize=(10, 5)
):
  """Hàm vẽ biểu đồ F-statistic và in p-value tương ứng (không chứa mốc 0.05)."""
  results = []
  for s in series:
    anova_res = anova_analisis(data, s, target)
    results.append({
        'Feature': s,
        'F_statistic': anova_res.statistic,
        'p_value': anova_res.pvalue,
    })

  # Sắp xếp theo F-statistic tăng dần để biến ảnh hưởng mạnh nhất nằm trên cùng
  df_res = pd.DataFrame(results).sort_values(by='F_statistic', ascending=True)

  fig, ax = plt.subplots(figsize=figsize)

  # Vẽ biểu đồ cột ngang bằng màu xanh đồng nhất
  bars = ax.barh(df_res['Feature'], df_res['F_statistic'], color='skyblue')

  # Hiển thị giá trị p-value trực tiếp bên cạnh mỗi cột
  max_f = max(df_res['F_statistic'])
  for bar, p_val in zip(bars, df_res['p_value']):
    width = bar.get_width()
    p_text = f'  p = {p_val:.2e}' if p_val < 0.001 else f'  p = {p_val:.4f}'
    ax.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        p_text,
        ha='left',
        va='center',
        fontsize=9,
        fontweight='bold',
    )

  # Trình bày giao diện biểu đồ
  ax.set_title(
      f'Mức độ ảnh hưởng của các biến phân loại tới {target} (ANOVA)',
      fontsize=12,
      fontweight='bold',
  )
  ax.set_xlabel(
      'F-test',
      fontsize=10,
  )
  ax.set_ylabel('Biến phân loại', fontsize=10)
  ax.set_xlim(0, max_f * 1.25)
  ax.grid(axis='x', linestyle='--', alpha=0.5)

  plt.tight_layout()
  plt.show()

def get_num_cat_var(data):
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()

    return numerical_cols, categorical_cols

def numeric_var_statistic(data: pd.DataFrame, numerical_cols: list):
  """Hàm tổng hợp phân tích đơn biến cho danh sách các biến số.

  Bao gồm: Thống kê mô tả (.describe), Skewness, Dashboard Boxplot và Dashboard
  Histplot (KDE).
  """
  # ==========================================
  # PHẦN 1: BÁO CÁO THỐNG KÊ SỐ HỌC & ĐỘ LỆCH
  # ==========================================
  print("=== THỐNG KÊ MÔ TẢ CÁC BIẾN SỐ ===")
  print(data[numerical_cols].describe().T)

  print("\n=== ĐỘ LỆCH (SKEWNESS) CỦA CÁC BIẾN SỐ ===")
  print(data[numerical_cols].skew())
  print("-" * 50)

  # ==========================================
  # PHẦN 2: DASHBOARD BOXPLOT (LOẠI BỎ KHỐI TRONG VÒNG LẶP DỄ LỖI INDENT)
  # ==========================================
  n_vars = len(numerical_cols)

  # Tự động tính số cột & hàng cho Boxplot
  if n_vars <= 4:
    n_cols_box = n_vars
  elif n_vars <= 8:
    n_cols_box = 4
  else:
    n_cols_box = 6

  n_rows_box = math.ceil(n_vars / n_cols_box)
  scale_box = max(1, n_rows_box / 2)

  fig1, axes_box = plt.subplots(
      n_rows_box, n_cols_box, figsize=(16 * scale_box, 9 * scale_box)
  )
  axes_box = [axes_box] if n_vars == 1 else axes_box.flatten()

  for i, col in enumerate(numerical_cols):
    sns.boxplot(
        y=data[col],
        ax=axes_box[i],
        color="lightgreen",
        flierprops={"markerfacecolor": "red", "marker": "o", "markersize": 4},
    )
    axes_box[i].set_title(f"Boxplot: {col}", fontsize=10, fontweight="bold")
    axes_box[i].set_ylabel("")
    axes_box[i].grid(axis="y", linestyle="--", alpha=0.6)

  # Ẩn các ô thừa Boxplot
  for j in range(len(numerical_cols), len(axes_box)):
    fig1.delaxes(axes_box[j])

  fig1.suptitle(
      "DASHBOARD PHÁT HIỆN OUTLIERS (BOXPLOT)", fontsize=14, fontweight="bold"
  )
  plt.tight_layout()
  plt.show()

  # ==========================================
  # PHẦN 3: DASHBOARD HISTPLOT + KDE (PHÂN BỐ)
  # ==========================================
  # Tự động tính số cột & hàng cho Histplot
  if n_vars <= 4:
    n_cols_hist = 2
  elif n_vars <= 9:
    n_cols_hist = 3
  else:
    n_cols_hist = 4

  n_rows_hist = math.ceil(n_vars / n_cols_hist)
  scale_hist = max(1, n_rows_hist / 2)

  fig2, axes_hist = plt.subplots(
      n_rows_hist, n_cols_hist, figsize=(16 * scale_hist, 9 * scale_hist)
  )
  axes_hist = [axes_hist] if n_vars == 1 else axes_hist.flatten()

  for i, col in enumerate(numerical_cols):
    sns.histplot(data[col], kde=True, bins=30, color="skyblue", ax=axes_hist[i])
    axes_hist[i].set_title(
        f"Phân bố: {col}", fontsize=10, fontweight="bold"
    )
    axes_hist[i].set_xlabel("")
    axes_hist[i].set_ylabel("Tần suất", fontsize=9)
    axes_hist[i].grid(axis="y", linestyle="--", alpha=0.6)

  # Ẩn các ô thừa Histplot
  for j in range(len(numerical_cols), len(axes_hist)):
    fig2.delaxes(axes_hist[j])

  fig2.suptitle(
      "DASHBOARD PHÂN BỐ DỮ LIỆU (HISTOGRAM & KDE)",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.show()

def categorical_var_statistic(data: pd.DataFrame, categorical_cols: list):
  """Hàm thống kê tần suất và vẽ Dashboard Countplot cho các biến phân loại."""
  # ==========================================
  # PHẦN 1: BÁO CÁO BẢNG TẦN SUẤT & TỶ LỆ (%)
  # ==========================================
  for col in categorical_cols:
    print(f"=== Tần suất cho biến: {col} ===")
    counts = data[col].value_counts()
    percents = data[col].value_counts(normalize=True) * 100

    freq_table = pd.DataFrame({"Số lượng": counts, "Tỷ lệ (%)": percents})
    print(freq_table)
    print("-" * 40)

  # ==========================================
  # PHẦN 2: TÍNH TOÁN KHUNG HÌNH DASHBOARD 16:9
  # ==========================================
  n_vars = len(categorical_cols)

  if n_vars <= 3:
    n_cols = n_vars
  elif n_vars <= 6:
    n_cols = 3
  else:
    n_cols = 4  # Tối đa 4 cột để nhãn tên nhóm không bị chèn ép

  n_rows = math.ceil(n_vars / n_cols)
  scale = max(1, n_rows / 2)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * scale, 9 * scale))

  # Chuyển axes thành mảng 1 chiều
  if n_vars == 1:
    axes = [axes]
  else:
    axes = axes.flatten()

  # ==========================================
  # PHẦN 3: VẼ BẢNG COUNTPLOT CHO TỪNG BIẾN
  # ==========================================
  for i, col in enumerate(categorical_cols):
    # Dùng order để sắp xếp các cột theo số lượng giảm dần
    sns.countplot(
        data=data,
        x=col,
        order=data[col].value_counts().index,
        palette="Set2",
        hue=col,
        legend=False,
        ax=axes[i],
    )

    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Số lượng", fontsize=9)
    axes[i].tick_params(axis="x", rotation=45)  # Xoay nhãn nhóm 45 độ
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

    # Thêm giá trị cụ thể trên đỉnh từng cột
    for p in axes[i].patches:
      height = p.get_height()
      if not math.isnan(height) and height > 0:
        axes[i].annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )

  # ==========================================
  # PHẦN 4: ẨN CÁC Ô THỪA VÀ HIỂN THỊ
  # ==========================================
  for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

  fig.suptitle(
      "DASHBOARD TẦN SUẤT CÁC BIẾN PHÂN LOẠI (COUNTPLOT)",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.show()