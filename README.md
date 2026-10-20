# 💻 PHÂN TÍCH CÁC YẾU TỐ CẤU HÌNH VÀ XÂY DỰNG MÔ HÌNH DỰ ĐOÁN GIÁ BÁN LAPTOP ĐÃ QUA SỬ DỤNG

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Package Manager](https://img.shields.io/badge/uv-fast_package_manager-purple)
![Tests](https://img.shields.io/badge/Tests-40%20Passed-brightgreen?logo=pytest)
![Linter](https://img.shields.io/badge/Linter-Ruff_Clean-blue?logo=ruff)
![Framework](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikit-learn)
![Course](https://img.shields.io/badge/Course-IE313_Data_Analysis-green)

Đồ án môn học: **Phân tích và Trực quan dữ liệu (IE313)** — Trường Đại học Công nghệ Thông tin (UIT)

---

## 📌 1. Giới thiệu Đề tài & Đặt vấn đề

Thị trường mua bán Laptop đã qua sử dụng hiện nay rất phát triển nhưng việc định giá sản phẩm thường mang tính cảm tính của người bán. Dự án này được thực hiện nhằm mục đích:

- **Phân tích các thuộc tính cấu hình** (RAM, CPU, GPU, bộ nhớ, kích thước màn hình, thương hiệu...) tác động mạnh nhất đến giá bán.
- **Xây dựng Pipeline tự động hóa** thực hiện Tiền xử lý dữ liệu → Phân tích EDA → Huấn luyện Mô hình Hồi quy nhằm dự đoán giá bán một cách minh bạch và chính xác.

---

## 📁 2. Cấu trúc Dự án (Directory Structure)

```text
ie313-laptop-price-prediction/
├── data/                            # Thư mục chứa dữ liệu dạng bảng
│   ├── raw/                         # Dữ liệu thô ban đầu (laptop_price.csv)
│   └── processed/                   # Dữ liệu sạch sau tiền xử lý (clean_laptop_price.csv)
├── src/                             # Thư mục mã nguồn chính (Python Package)
│   ├── __init__.py                  # Đánh dấu Python Package
│   ├── config.py                    # Cấu hình đường dẫn, hằng số, random_state
│   ├── data_loader.py               # Đọc & kiểm tra thông tin dữ liệu thô
│   ├── preprocessor.py              # Tiền xử lý (xử lý khuyết, One-Hot Encoding, Z-score)
│   ├── eda.py                       # Trực quan hóa & Kiểm định ANOVA / Pearson Correlation
│   ├── model_pipeline.py            # Scikit-Learn Pipeline (Poly Features + Linear Regression)
│   └── utils.py                     # Các hàm phụ trợ (trích xuất cấu hình CPU, GPU, RAM, Memory)
├── tests/                           # Thư mục kiểm thử tự động (Unit & Integration Tests)
│   ├── __init__.py                  # Đánh dấu package kiểm thử
│   ├── conftest.py                  # Fixtures dùng chung & cấu hình headless matplotlib (Agg)
│   ├── test_config.py               # Kiểm thử đường dẫn và tham số cấu hình
│   ├── test_data_loader.py          # Kiểm thử nạp dữ liệu thô và xử lý ngoại lệ
│   ├── test_eda.py                  # Kiểm thử thống kê, tương quan Pearson, kiểm định ANOVA
│   ├── test_integration.py          # Kiểm thử tích hợp toàn bộ pipeline từ đầu tới cuối
│   ├── test_model_pipeline.py       # Kiểm thử huấn luyện mô hình hồi quy
│   ├── test_preprocessor.py         # Kiểm thử làm sạch, tiền xử lý và lọc outliers
│   └── test_utils.py                # Kiểm thử các hàm tiện ích trích xuất CPU/GPU/Ổ cứng
├── reports/                         # Báo cáo & Slide thuyết trình
│   ├── Nhom5_Bao_cao.docx           # Báo cáo Word (theo mẫu Template_IE313.docx)
│   ├── Nhom5_Bao_cao.pdf            # File PDF tương ứng
│   ├── Nhom5_Slide.pptx             # Slide thuyết trình đồ án
│   └── img/                         # Các biểu đồ trực quan xuất từ quá trình EDA
├── .gitignore                       # Bỏ qua .venv, __pycache__, .ruff_cache, .pytest_cache
├── main.py                          # Entry point chạy toàn bộ quy trình từ Command Line
├── demo.ipynb                       # Notebook dùng để chạy trình diễn & hiển thị biểu đồ
├── pyproject.toml                   # Cấu hình dự án, dependencies và pytest do uv quản lý
├── uv.lock                          # File khóa phiên bản thư viện chính xác của uv
├── requirements.txt                 # File xuất phụ thuộc dự phòng cho pip
├── PROJECT_COMPLETION_REPORT.md     # Báo cáo đánh giá mức độ hoàn thiện đồ án
└── README.md                        # Hướng dẫn cài đặt, chạy & kiểm thử dự án
```

---

## 🛠️ 3. Yêu cầu & Hướng dẫn Cài đặt Môi trường (`uv`)

Dự án sử dụng trình quản lý môi trường siêu tốc **`uv`** (viết bằng Rust).

### Bước 1: Clone Repository

```bash
git clone https://github.com/vanthom04/ie313-laptop-price-prediction.git
cd ie313-laptop-price-prediction
```

### Bước 2: Khởi tạo & Đồng bộ Môi trường với `uv`

```bash
# Tạo và kích hoạt môi trường ảo
uv venv
source .venv/bin/activate  # Trên macOS/Linux
# Hoặc trên Windows: .venv\Scripts\activate

# Đồng bộ tự động tất cả thư viện từ pyproject.toml
uv sync
```

*(Phương án dự phòng nếu chạy trên máy chỉ có `pip` truyền thống: `pip install -r requirements.txt`)*

---

## 🚀 4. Hướng dẫn Chạy Chương trình & Kiểm thử

### Option 1: Chạy quy trình tự động hoàn chỉnh từ Terminal

```bash
uv run python main.py
```

### Option 2: Mở Jupyter Notebook để xem trực quan biểu đồ

```bash
uv run jupyter lab
# Sau đó mở file demo.ipynb để xem biểu đồ và kết quả chi tiết từng bước
```

### Option 3: Chạy bộ Kiểm thử Tự động Toàn diện (`pytest`)

Dự án tích hợp bộ kiểm thử gồm 40 bài test (Unit Tests & Integration Tests) bao phủ toàn bộ các mô-đun:

```bash
uv run pytest -v
```

### Option 4: Kiểm tra Chuẩn Mã nguồn & Linting (`ruff`)

```bash
uv run ruff check .
```

---

## ⚙️ 5. Quy trình Kỹ thuật (Data Pipeline Workflow)

1. **Data Loading (`src/data_loader.py`):** Kiểm tra kiểu dữ liệu, kích thước tập dữ liệu và tính toàn vẹn của 13 thuộc tính dữ liệu thô.
2. **Data Preprocessing (`src/preprocessor.py`):**
   - Xử lý giá trị khuyết (`NaN`) bằng Median (cho biến số) hoặc Mode (cho biến phân loại).
   - Trích xuất đặc trưng phần cứng: Tần số CPU (`Cpu_Speed`), phân loại CPU (`Cpu_Brand`), dung lượng ổ cứng tách theo phân loại (`SSD`, `HDD`, `Flash_Storage`, `Hybrid`), loại card đồ họa (`Gpu_Brand`, `Gpu_Type`), độ phân giải màn hình (`Touchscreen`, `Ips`).
   - Lọc bỏ ngoại lệ (Outliers) dựa trên khoảng tứ phân vị (IQR).
   - Mã hóa biến phân loại bằng **One-Hot Encoding** (`drop_first=True`).
   - Chuẩn hóa Z-score (`StandardScaler`) đưa các biến số về cùng thang đo chuẩn.
3. **EDA & Feature Selection (`src/eda.py`):**
   - Biểu diễn phân bố giá và phát hiện ngoại lệ bằng Dashboard Boxplot & Histplot.
   - Vẽ Heatmap tương quan Pearson giữa các biến số với biến mục tiêu (`Price_euros`), lọc các biến có $|corr| > 0.2$.
   - Lọc thuộc tính phân loại quan trọng qua kiểm định thống kê **One-way ANOVA** ($p < 0.05$).
4. **Model Development (`src/model_pipeline.py`):**
   - Đóng gói quy trình trong `sklearn.pipeline.Pipeline` kết hợp **PolynomialFeatures** (mặc định bậc 1 để đảm bảo tính tổng quát hóa cao và tránh bùng nổ số chiều) và **LinearRegression**.
   - Chia tập Train/Test theo tỷ lệ 80/20 (`train_test_split`, `random_state=42`).
5. **Model Evaluation:** Đánh giá độ chính xác qua chỉ số $R^2$ Score trên cả tập Train ($R^2 \approx 0.7916$) và Test ($R^2 \approx 0.7910$), đảm bảo mô hình không bị quá khớp (overfitting).

---

## 👥 6. Phân công Nhiệm vụ (Team Members)

| STT   | Thành viên         | Vai trò                          | Công việc phụ trách                                                                                                             |
| :---- | :----------------- | :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **1** | **Chu Văn Thơm**  | Project Lead & System Integrator | Quản lý môi trường `uv`, `config.py`, `main.py`, `demo.ipynb`, tests tự động, tổng hợp Báo cáo/Slide, thuyết trình Demo code.    |
| **2** | **Trần Ngọc Tâm** | Data & Preprocessing Lead        | Thu thập dữ liệu thô, viết `data_loader.py`, `preprocessor.py`, viết phần Dữ liệu trong Báo cáo/Slide, biên tập Video MS Teams. |
| **3** | **Hồ Thiên Phúc** | EDA & Modeling Lead              | Viết `eda.py`, `model_pipeline.py` (ANOVA, Pearson, Pipeline Mô hình), viết phần EDA & Mô hình trong Báo cáo/Slide.             |

---

## 🛡️ 7. Cam kết Minh bạch (Transparency Statement)

Bộ dữ liệu sử dụng trong dự án được tham khảo và thu thập công khai từ nguồn uy tín (Kaggle/UCI Datasets). Toàn bộ mã nguồn, nội dung báo cáo và slide thuyết trình đều do các thành viên trong nhóm tự thực hiện, tuân thủ nghiêm ngặt quy định minh bạch học thuật của môn học IE313.
