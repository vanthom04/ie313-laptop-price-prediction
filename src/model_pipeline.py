# Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt


def train_data_pipeline(data, degree=1):
    y_data = data["Price_euros"]
    x_data = data.drop(["Price_euros"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.8, random_state=0
    )

    # Đóng gói Pipeline
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression()),
    ])

    # Fit pipeline
    pipe.fit(x_train, y_train)

    # Dự đoán trên tập test
    y_hat = pipe.predict(x_test)

    # Sửa lỗi truyền biến và thêm nhãn (label) cho 2 đường
    plt.figure(figsize=(8, 5))
    ax1 = sns.kdeplot(y_test, color='r', label='Giá thực tế (Actual Price)')
    sns.kdeplot(y_hat, color='b', label='Giá dự đoán (Predicted Price)', ax=ax1)

    # Hiển thị chú thích nhãn và các thông tin biểu đồ
    plt.title(
        f'So sánh Giá thực tế vs Giá dự đoán (Degree = {degree})',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Price_euros')
    plt.ylabel('Mật độ (Density)')
    plt.legend(loc='upper right')  # Hiển thị hộp chú thích nhãn ở góc trên bên phải
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    return pipe.score(x_train, y_train), pipe.score(x_test, y_test)