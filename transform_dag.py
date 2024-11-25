import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import joblib
import pickle


def process_car_data_1(df):
    # Фильтруем строки, где заголовок начинается с 'Продажа'
    df = df[df['title'].str.startswith('Продажа')]

    # Извлечение модели, года и местоположения из заголовка
    pattern = r'Продажа (.*?), (\d{4}) год в (.*?) -'
    df[['model', 'year', 'location']] = df['title'].str.extract(pattern)
    df.drop(columns=['title'], inplace=True)

    # Извлечение типа и объёма двигателя
    pattern = r'(\w+), (\d+\.\d+ л)'
    df[['type_engine', 'volume_engine']] = df['engine'].str.extract(pattern)

    # Удаление ненужных столбцов
    df.drop(columns=['engine', 'description'], inplace=True)

    # Обработка столбца 'power'
    df['power'] = df['power'].str.split(',').str[0]
    df['location'] = 'в ' + df['location']
    
    df['year'] = pd.to_numeric(df['year'])                                         ### Год приводим к числовому значению


    # Упорядочивание столбцов
    new_order = [
        'model', 'year', 'price', 'location', 'type_engine', 'volume_engine', 
        'power', 'transmission', 'drive', 'body_type', 'color', 'mileage', 
        'steering', 'generation', 'complectation', 'additional_info', 'image_url', 'listing_url'
    ]

    # Структура по новому порядку и удаление пустых значений в столбце 'model'
    df = df[new_order]
    df = df[df['model'].notna() & (df['model'] != '')]
    df.reset_index(drop=True, inplace=True)

    return df

def preprocess_data(df):
    # Удаляем ненужные колонки, но оставляем 'listing_url' и 'image_url'
    df = df.drop(columns=['Unnamed: 0', 'generation', 'complectation', 'additional_info'], errors='ignore')
    
    # Переименовываем исходную колонку 'model' для сохранения её содержимого
    df.rename(columns={'model': 'full_model'}, inplace=True)
    
    # Разделяем 'full_model' на 'brand' и 'model_name'
    df[['brand', 'model']] = df['full_model'].str.split(n=1, expand=True)
    
    # Удаляем колонку 'full_model', если она больше не нужна
    df.drop(columns=['full_model'], inplace=True)
    
    # Далее ваш код без изменений
    # Обработка 'price'
    df['price'] = df['price'].str.replace('\xa0', '', regex=False).str.replace('₽', '', regex=False).str.replace(' ', '', regex=False).astype(int)
    
    # Обработка 'volume_engine'
    df['volume_engine'] = df['volume_engine'].str.replace('\xa0', '', regex=False).str.replace(' л', '', regex=False).astype(float)
    df['volume_engine'] = df['volume_engine'].apply(lambda x: x if x <= 7 else 7)
    
    # Обработка 'power'
    df['power'] = df['power'].replace('Нет данных', np.nan)
    df = df.dropna(subset=['power'])
    df['power'] = df['power'].str.replace('\xa0', '', regex=False).str.replace('л.с.', '', regex=False).astype(float)
    df = df[(df['power'] >= 40) & (df['power'] <= 600)]
    
    # Обработка 'body_type'
    df['body_type'] = df['body_type'].replace({
        'хэтчбек 5 дв.': 'хэтчбек',
        'хэтчбек 3 дв.': 'хэтчбек',
        'джип/suv 5 дв.': 'джип/SUV',
        'джип/suv 3 дв.': 'джип/SUV',
        'джип/suv': 'джип/SUV',
        'джип/SUV': 'джип/SUV',
    })
    
    # Обработка 'mileage'
    df = df[df['mileage'] != 'Нет данных']
    df['mileage'] = df['mileage'].str.strip().str.lower()
    df = df[~df['mileage'].str.contains('безпробегапорф')]
    df['mileage'] = df['mileage'].replace({'новый автомобиль': '0'})
    df['mileage'] = df['mileage'].str.replace(r'\s+', '', regex=True).str.extract(r'(\d+)').astype(float)
    df = df.dropna(subset=['mileage'])
    
    # Обработка 'transmission'
    df['transmission'] = df['transmission'].replace('АКПП', 'автомат').fillna('Нет данных')
    df = df[df['transmission'] != 'Нет данных']
    
    # Обработка 'drive'
    df['drive'] = df['drive'].fillna('передний')
    
    # Обработка 'type_engine'
    df['type_engine'] = df['type_engine'].fillna('бензин')
    
    # Обработка 'color'
    df['color'] = df['color'].replace({'Нет данных': 'черный', 'чёрный': 'черный'})
    
    # Обработка 'steering'
    df = df[df['steering'] != 'Нет данных']
    
    # Сброс индексов
    df.reset_index(drop=True, inplace=True)
    
    return df

def create_preprocessor():
    # Определяем числовые и категориальные признаки
    numerical_features = ['year', 'volume_engine', 'power', 'mileage']
    categorical_features = ['brand', 'model', 'location', 'type_engine', 'transmission', 'drive', 'body_type', 'steering', 'color']
    
    # Трансформер для числовых признаков
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Трансформер для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Объединение в ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor


def create_pipeline(preprocessor):
    # Настройка модели CatBoost
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    )
    
    # Создание Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

# Загрузка данных
#df = pd.read_csv('cars_filtered.csv', encoding='utf-8')
# df = pd.read_csv('Cars.csv', encoding='utf-8')

df = pd.read_csv('/home/aleksey/DS_bootcamp/air_test_2/Cars.csv', encoding='utf-8')

df_new = pd.read_csv('/home/aleksey/DS_bootcamp/air_test_2/airflow.csv', encoding='utf-8')

df = pd.concat([df, df_new])

# df.to_csv('Cars.csv.csv', index=False)

# Предобработка данных

df = process_car_data_1(df)                          ### Добавлена функция, которая обрабатывает от Cars.csv до cars.filtered.csv

df_processed = preprocess_data(df)

# Разделение на признаки и целевую переменную
X = df_processed.drop(['price', 'listing_url', 'image_url'], axis=1)
y = df_processed['price']

# Разделение на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание препроцессора и Pipeline
preprocessor = create_preprocessor()
pipeline = create_pipeline(preprocessor)

# Обучение модели
pipeline.fit(X_train, y_train)

# # Сохранение модели с использованием pickle вариант 3
# with open('model_pipeline.pkl', 'wb') as f:
#     pickle.dump(pipeline, f)

# Сохранение Pipeline и модели вариан 2
joblib.dump(pipeline, 'model_pipeline.pkl')
print("Pipeline сохранён как 'model_pipeline.pkl'")

# # Сохранение модели вариант 1
# catboost_model = pipeline.named_steps['model']  # Извлекаем модель из pipeline
# catboost_model.save_model('trained_model.cbm')  # Сохраняем модель
# print("Модель сохранена как 'trained_model.cbm'")

# Оценка модели
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
y_pred = pipeline.predict(X_valid)

# Рассчитываем метрики
mae = mean_absolute_error(y_valid, y_pred)
mape = mean_absolute_percentage_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

# Выводим метрики
with open('/home/aleksey/DS_bootcamp/air_test_2/model_metrics.txt', 'w') as f:
    f.write(f'Mean Absolute Error: {mae}\n')
    f.write(f'Mean Absolute Percentage Error: {mape * 100:.2f}%\n')
    f.write(f'R² Score: {r2:.2f}\n')

    
 # process_data(df)


default_args = {
    'owner': 'AutoChoice'
}


dag = DAG(
    'car_data_pipline',
    default_args=default_args,
    description='Pipeline for processing car data',
    start_date=datetime(2024, 11, 20),
    schedule_interval='14 21 * * *',
    catchup=False,
)

# Определяем задачу обработки данных
process_task = PythonOperator(
    task_id='process_car_data',
    python_callable=create_pipeline,
    op_kwargs={'df': df},  # Передаем DataFrame как аргумент
    provide_context=True,
    dag=dag,
)

process_task
