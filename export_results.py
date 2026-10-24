import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không mở cửa sổ, chỉ lưu file ảnh

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE
from src.model_pipeline import train_data_pipeline

OUT_DIR = Path("kde_out")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(PROCESSED_DATA_PATH)
print(
    f"Dữ liệu: {df.shape[0]} dòng, {df.shape[1] - 1} đặc trưng | "
    f"random_state={RANDOM_STATE}, test_size={TEST_SIZE}"
)

for degree in (1, 2, 3):
    # train_data_pipeline gọi plt.show(): chuyển thành lưu file
    plt.show = lambda d=degree: (
        plt.savefig(OUT_DIR / f"kde_bac_{d}.png", dpi=200, bbox_inches="tight"),
        plt.close(),
    )
    r2_train, r2_test = train_data_pipeline(df, degree=degree)
    print(
        f"Bậc {degree}: R2 train = {r2_train:.4f} | R2 test = {r2_test:.4f} "
        f"| chênh lệch = {r2_train - r2_test:.4f}"
    )
