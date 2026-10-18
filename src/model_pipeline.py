# Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

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

    return pipe.score(x_train, y_train), pipe.score(x_test, y_test)

# def train_data(data, degree):
#     y_data = data[["Price_euros"]]
#     x_data = data.drop(["Price_euros"], axis=1)

#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size= 0.8, random_state=0)

#     ss = StandardScaler()
#     x_train_ss = ss.fit_transform(x_train)
#     x_test_ss = ss.transform(x_test)

#     pf = PolynomialFeatures(degree=degree, include_bias=False)
#     x_train_pf = pf.fit_transform(x_train_ss)
#     x_test_pf = pf.fit_transform(x_test_ss)

#     m2 = LinearRegression()
#     m2.fit(x_train_pf, y_train)

#     y_hat = m2.predict(x_train_pf).flatten()

#     ax1 = sns.kdeplot(data = y_test, x="Price_euros")
#     sns.kdeplot(x=y_hat, ax=ax1)

#     return m2.score(x_train_pf, y_train), m2.score(x_test_pf, y_test)