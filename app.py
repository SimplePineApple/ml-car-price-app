# при написании этого файла я сверялась с ChatGPT, так как было много ошибок
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# базовые настройки страницы
st.set_page_config(page_title="Car price app", layout="wide")


# загрузка данных и артефактов
@st.cache_data
def load_train_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    return pd.read_csv(url)


@st.cache_resource
def load_artifacts():
    with open("car_price_artifacts.pkl", "rb") as f:
        return pickle.load(f)


df_train = load_train_data()
art = load_artifacts()

model = art["model"]
scaler = art["scaler"]
ohe = art["ohe"]
num_cols = art["num_cols"]
cat_cols = art["cat_cols"]


# небольшая функция для подготовки признаков
def prepare_features(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Ожидает на входе DataFrame, в котором есть колонки num_cols + cat_cols.
    Возвращает матрицу признаков в том виде, в каком их ждёт модель.
    """
    x_num = df_raw[num_cols]
    x_num_scaled = scaler.transform(x_num)
    x_cat = df_raw[cat_cols]
    x_cat_ohe = ohe.transform(x_cat)
    x_full = np.hstack([x_num_scaled, x_cat_ohe])
    return x_full


# интерфейс приложения
st.title("Приложение по предсказанию цены автомобиля")

mode = st.sidebar.radio("Режим работы:", ["EDA", "Предсказания", "Веса модели"])

# блок EDA
if mode == "EDA":
    st.header("Исследование данных (EDA)")
    st.write("Первые строки обучающего датасета:")
    st.dataframe(df_train.head())

    # выбор числового признака для гистограммы
    num_column = st.selectbox(
        "Выберите числовой признак для гистограммы",
        df_train.select_dtypes("number").columns,
    )

    fig, ax = plt.subplots()
    ax.hist(df_train[num_column].dropna(), bins=30)
    ax.set_title(f"Гистограмма {num_column}")
    ax.set_xlabel(num_column)
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    st.subheader("Описательная статистика выбранного признака")
    st.write(df_train[num_column].describe())

# блок предсказаний
if mode == "Предсказания":
    st.header("Предсказания модели")
    st.markdown(
        """
        **Вариант 1. Загрузить CSV с признаками объектов.**  
        В файле должны быть те же колонки, что использовались при обучении:
        - числовые: из `num_cols`;
        - категориальные: из `cat_cols`.
        """
    )

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    # предсказания по csv
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:", new_df.head())
        try:
            X_new = prepare_features(new_df)
            preds = model.predict(X_new)
            new_df["prediction"] = preds
            st.write("Результат с предсказаниями (первые строки):")
            st.dataframe(new_df.head())
        except Exception as e:
            st.error(
                "Не удалось применить модель. "
                "Проверьте, что в CSV есть все нужные колонки.\n"
                f"Техническая ошибка: {e}"
            )

    st.markdown("---")
    st.subheader("Вариант 2. Ввести один объект вручную")

    # числовые признаки вводим через number_input
    input_num = {}
    for col in num_cols:
        # просто стартовое значение, чтобы поле не было пустым
        input_num[col] = st.number_input(f"{col}", value=0.0)

    # категориальные признаки – через selectbox / текст
    input_cat = {}
    for col in cat_cols:
        if col in df_train.columns:
            options = sorted(df_train[col].dropna().unique().tolist())
            input_cat[col] = st.selectbox(f"{col}", options)
        else:
            input_cat[col] = st.text_input(f"{col}", "")

    # кнопка ВНЕ цикла по cat_cols
    if st.button("Сделать предсказание", key="predict_one_button"):
        one_row = {}
        for col in num_cols:
            one_row[col] = input_num[col]
        for col in cat_cols:
            one_row[col] = input_cat[col]

        df_one = pd.DataFrame([one_row])
        try:
            X_one = prepare_features(df_one)
            pred_one = model.predict(X_one)[0]
            st.success(f"Предсказанная цена автомобиля: {pred_one:,.0f}")
        except Exception as e:
            st.error(f"Ошибка при подготовке признаков или предсказании: {e}")


# блок весов модели
if mode == "Веса модели":
    st.header("Веса (коэффициенты) модели")
    # для Ridge: линейная модель с coef_
    coefs = model.coef_.flatten()

    # имена фич после ohe
    try:
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
    except TypeError:
        # на случай старой версии sklearn
        cat_feature_names = ohe.get_feature_names(cat_cols)

    feature_names = list(num_cols) + list(cat_feature_names)

    weights_df = pd.DataFrame({"feature": feature_names, "weight": coefs}).sort_values(
        "weight", ascending=False
    )

    st.write("Таблица с весами признаков:")
    st.dataframe(weights_df)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(weights_df["feature"], weights_df["weight"])
    ax.set_xlabel("Вес")
    ax.set_ylabel("Признак")
    ax.invert_yaxis()
    st.pyplot(fig)
