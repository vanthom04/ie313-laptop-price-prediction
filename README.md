# 💻 PHÂN TÍCH CÁC YẾU TỐ CẤU HÌNH VÀ XÂY DỰNG MÔ HÌNH DỰ ĐOÁN GIÁ BÁN LAPTOP ĐÃ QUA SỬ DỤNG

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Package Manager](https://img.shields.io/badge/uv-fast_package_manager-purple)
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
├── data/                       # Thư mục chứa dữ liệu dạng bảng
│   ├── raw/                    # Dữ liệu thô ban đầu
│   └── processed/              # Dữ liệu đã qua tiền xử lý/làm sạch (nếu có)
├── src/                        # Thư mục mã nguồn chính (Python Package)
│   ├── __init__.py             # Đánh dấu Python Package
│   ├── config.py               # Cấu hình đường dẫn, hằng số, random_state
│   ├── data_loader.py          # Đọc & kiểm tra thông tin dữ liệu thô
│   ├── preprocessor.py         # Tiền xử lý (xử lý khuyết, One-Hot Encoding, Z-score)
│   ├── eda.py                  # Trực quan hóa & Kiểm định ANOVA / Pearson Correlation
│   ├── model_pipeline.py       # Scikit-Learn Pipeline (Poly Features + Linear Regression)
│   └── utils.py                # Các hàm phụ trợ (vẽ hình, lưu kết quả, logging)
├── reports/                    # Báo cáo & Slide thuyết trình
│   ├── NhomX_Bao_cao.docx      # Báo cáo Word (theo mẫu Template_IE313.docx)
│   ├── NhomX_Bao_cao.pdf       # File PDF tương ứng
│   └── NhomX_Slide.pptx        # Slide thuyết trình đồ án
├── .gitignore                  # Bỏ qua .venv, __pycache__, .ipynb_checkpoints
├── main.py                     # Entry point chạy toàn bộ quy trình từ Command Line
├── notebook-demo.ipynb         # Notebook dùng để chạy trình diễn & hiển thị biểu đồ
├── pyproject.toml              # File cấu hình dự án & dependencies chính do uv quản lý
├── uv.lock                     # File khóa phiên bản thư viện chính xác của uv
├── requirements.txt            # File xuất dự phòng cho pip
└── README.md                   # Hướng dẫn cài đặt & chạy dự án
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

_(Phương án dự phòng nếu chạy trên máy chỉ có `pip` truyền thống: `pip install -r requirements.txt`)_

---

## 🚀 4. Hướng dẫn Chạy Chương trình

### Option 1: Chạy quy trình tự động hoàn chỉnh từ Terminal

```bash
uv run python main.py
```

### Option 2: Mở Jupyter Notebook để xem trực quan biểu đồ

```bash
uv run jupyter lab
# Sau đó mở file demo.ipynb để xem biểu đồ và kết quả chi tiết từng bước
```

---

## ⚙️ 5. Quy trình Kỹ thuật (Data Pipeline Workflow)

1. **Data Loading (`src/data_loader.py`):** Kiểm tra kiểu dữ liệu, kích thước tập dữ liệu và tính toàn vẹn của dữ liệu thô.
2. **Data Preprocessing (`src/preprocessor.py`):**
   - Xử lý giá trị khuyết (`NaN`) bằng Median hoặc Mode.
   - Mã hóa biến phân loại (`Brand`, `CPU`, `OS`) bằng **One-Hot Encoding**.
   - Chuẩn hóa Z-score (`StandardScaler`) đưa các biến số về cùng thang đo.
3. **EDA & Feature Selection (`src/eda.py`):**
   - Biểu diễn phân bố giá và phát hiện ngoại lệ bằng Boxplot & Histplot.
   - Vẽ Heatmap tương quan Pearson giữa các biến số với giá bán.
   - Lọc thuộc tính phân loại quan trọng qua kiểm định thống kê **ANOVA** (\\(p < 0.05\\)).
4. **Model Development (`src/model_pipeline.py`):**
   - Đóng gói quy trình trong `sklearn.pipeline.Pipeline` kết hợp **PolynomialFeatures** (bậc 2) và **LinearRegression**.
   - Chia tập Train/Test theo tỷ lệ 80/20 (`train_test_split`).
5. **Model Evaluation:** Đánh giá độ chính xác qua chỉ số \\(R^2\\) Score và Mean Squared Error (MSE) trên cả tập Train và Test.

---

## 👥 6. Phân công Nhiệm vụ (Team Members)

| STT   | Thành viên         | Vai trò                          | Công việc phụ trách                                                                                                             |
| :---- | :----------------- | :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **1** | **Chu Văn Thơm**  | Project Lead & System Integrator | Quản lý môi trường `uv`, `config.py`, `main.py`, `demo.ipynb`, tổng hợp Báo cáo/Slide, thuyết trình Demo code.                  |
| **2** | **Trần Ngọc Tâm** | Data & Preprocessing Lead        | Thu thập dữ liệu thô, viết `data_loader.py`, `preprocessor.py`, viết phần Dữ liệu trong Báo cáo/Slide, biên tập Video MS Teams. |
| **3** | **Hồ Thiên Phúc** | EDA & Modeling Lead              | Viết `eda.py`, `model_pipeline.py` (ANOVA, Pearson, Pipeline Mô hình), viết phần EDA & Mô hình trong Báo cáo/Slide.             |

---

## 🛡️ 7. Cam kết Minh bạch (Transparency Statement)

Bộ dữ liệu sử dụng trong dự án được tham khảo và thu thập công khai từ nguồn uy tín (Kaggle/UCI Datasets). Toàn bộ mã nguồn, nội dung báo cáo và slide thuyết trình đều do các thành viên trong nhóm tự thực hiện, tuân thủ nghiêm ngặt quy định minh bạch học thuật của môn học IE313.
