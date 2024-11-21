import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from category_encoders import TargetEncoder
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta


df = pd.read_csv('/home/aleksey/DS_bootcamp/ФИНПРОЕКТ/датасеты/Cars.csv')

df_new = pd.read_csv('/home/aleksey/DS_bootcamp/ФИНПРОЕКТ/airflow.csv')

df = pd.concat([df, df_new])

# df.to_csv('Cars.csv.csv', index=False)

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


def process_car_data_2(df):
    # Удаление ненужной колонки
    # df = df.drop(columns=['Unnamed: 0'])
    
    # Разделение модели на марку и модель
    if 'model' in df.columns:
        # Разделение модели на марку и модель; добавляем дополнительные аргументы для обработки некорректных данных
        split_models = df['model'].str.split(n=1, expand=True)
        
        # Переименовываем созданные столбцы
        split_models.columns = ['brand', 'model']
        
        # Убедимся, что столбцы соответствуют длине DataFrame и присваиваем только существующие данные
        df[['brand', 'model']] = split_models

    # Обработка цены
    df['price'] = df['price'].str.replace('\xa0', '', regex=False)  # Удаляем неразрывные пробелы
    df['price'] = df['price'].str.replace('₽', '', regex=False)   # Удаляем символ ₽
    df['price'] = df['price'].str.replace(' ', '', regex=False)    # Удаляем обычные пробелы
    df['price'] = df['price'].astype(int)

    # Обработка объема двигателя
    df['volume_engine'] = df['volume_engine'].str.replace('\xa0', '', regex=False)  # Удаляем неразрывные пробелы
    df['volume_engine'] = df['volume_engine'].str.replace(' л', '', regex=False)   # Удаляем "л"
    df['volume_engine'] = pd.to_numeric(df['volume_engine'], errors='coerce')

    # Обработка мощности
    df['power'] = df['power'].str.replace('\xa0', '', regex=False)  # Удаляем неразрывные пробелы
    df['power'] = df['power'].str.replace('л.с.', '', regex=False)  # Удаляем "л.с."
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df = df[(df['power'] >= 40) & (df['power'] <= 600)]

    # Обработка типа кузова
    df['body_type'] = df['body_type'].replace({
        'хэтчбек 5 дв.': 'хэтчбек',
        'хэтчбек 3 дв.': 'хэтчбек',
        'джип/suv 5 дв.': 'джип/suv',
        'джип/suv 3 дв.': 'джип/suv'
    })

    # Фильтрация по пробегу
    df = df[df['mileage'] != 'Нет данных']
    df['mileage'] = df['mileage'].str.strip().str.lower()

    # Удаление строк с "без пробега по рф"
    df = df[df['mileage'] != "безпробегапорф"]

    # Заменяем текстовые описания на числовые значения
    df['mileage'] = df['mileage'].replace({
        'новый автомобиль': '0.0'  # Новый автомобиль = 0 км
    })

    # Оставление только числовой части пробега
    df['mileage'] = df['mileage'].str.replace(r'\s+', '', regex=True)  # Удаляем пробелы
    df['mileage'] = df['mileage'].str.extract(r'(\d+)')                # Извлекаем только числовые значения
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    # Удаление колонки 'generation'
    df = df.drop(columns=['generation'], errors='ignore')  # Используем errors='ignore', чтобы избежать ошибок, если колонки нет

    # Перемещение последней колонки в начало
    last_column = df.columns[-1] if not df.columns.empty else None
    if last_column:
        df = df[[last_column] + [col for col in df.columns if col != last_column]]

    return df


def process_car_data_3(df):
    # Заполняем пропущенные значения в 'type_engine' значением 'Нет данных'
    df['type_engine'] = df['type_engine'].fillna('Нет данных')
    
    # Вычисляем модальные значения для 'type_engine'
    mode_values = df.groupby(['model', 'year'])['type_engine'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else 'бензин'
    )
    
    # Заменяем "Нет данных" на рассчитанные значения моды
    df.loc[df['type_engine'] == "Нет данных", 'type_engine'] = mode_values
    df['type_engine'] = df['type_engine'].replace("Нет данных", "бензин")
    
    # Ограничиваем значение 'volume_engine' до 7
    df['volume_engine'] = df['volume_engine'].apply(lambda x: x if x <= 7 else 7)
    
    # Заменяем 'АКПП' на 'автомат' в 'transmission'
    df['transmission'] = df['transmission'].replace('АКПП', 'автомат')
    
    # Вычисляем модальные значения для 'transmission'
    mode_values = df.groupby(['model', 'year'])['transmission'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else 'Нет данных'
    )
    
    # Заменяем "Нет данных" на рассчитанные значения моды
    df.loc[df['transmission'] == "Нет данных", 'transmission'] = mode_values
    
    # Удаляем строки с "Нет данных" в 'transmission'
    df = df[df['transmission'] != 'Нет данных']
    
    # Вычисляем модальные значения для 'drive'
    mode_values = df.groupby(['model', 'year'])['drive'].transform(
        lambda x: x.mode()[0] if not x.mode().empty else 'передний'
    )
    
    # Заменяем "Нет данных" на рассчитанные значения моды
    df.loc[df['drive'] == "Нет данных", 'drive'] = mode_values
    df['drive'] = df['drive'].replace("Нет данных", "передний")
    
    # Заменяем "Нет данных" в 'color' на "чёрный"
    df['color'] = df['color'].replace("Нет данных", "черный")
    
    # Удаляем строки, где 'mileage' является NaN
    df = df.dropna(subset=['mileage'])
    
    # Удаляем строку с индексом 46889, если существует
    # if 46889 in df.index:
    #     df = df.drop(index=46889)
    
    # Удаляем строки, где 'steering' является "Нет данных"
    df = df[df['steering'] != "Нет данных"]
    
    return df


def process_car_data_4(df):
    model_to_body_type = {
        # Audi
        "A3": "седан", "A4": "седан", "A5": "купе", "A6": "седан",
        "Q3": "джип/SUV", "Q5": "джип/SUV", "Q7": "джип/SUV",
        # BMW
        "5-Series": "седан", "3-Series": "седан", "7-Series": "седан",
        "X5": "джип/SUV", "X6": "джип/SUV", "X3": "джип/SUV", "X1": "джип/SUV",
        # Changan
        "CS35 Plus": "джип/SUV", "CS55 Plus": "джип/SUV", "UNI-K": "джип/SUV",
        "UNI-S": "джип/SUV", "UNI-T": "джип/SUV", "UNI-V": "лифтбек", "Alsvin": "седан",
        # Chevrolet
        "Aveo": "седан", "Captiva": "джип/SUV", "Cruze": "седан", "Lacetti": "хэтчбек",
        "Lanos": "седан", "Niva": "джип/SUV", "Spark": "хэтчбек",
        # Hyundai
        "Accent": "седан", "Solaris": "седан", "Getz": "хэтчбек", "Sonata": "седан",
        "Tucson": "джип/SUV", "Creta": "джип/SUV", "Elantra": "седан", "Santa Fe": "джип/SUV",
        # Mitsubishi
        "Lancer": "седан", "Pajero Sport": "джип/SUV", "RVR": "хэтчбек", "Galant": "седан",
        "Delica": "минивэн", "ASX": "джип/SUV", "Outlander": "джип/SUV",
        # Subaru
        "Forester": "джип/SUV", "Impreza": "седан", "Legacy": "седан", "Legacy B4": "седан",
        "Levorg": "универсал", "Outback": "универсал", "XV": "джип/SUV",
        # Lada
        "Granta": "седан", "Priora": "седан", "Vesta": "седан", "2114": "хэтчбек",
        "2115 Самара": "седан", "Largus": "универсал", "4x4 2121 Нива": "джип/SUV",
        # Kia
        "Ceed": "хэтчбек", "Cerato": "седан", "Optima": "седан", "Rio": "седан",
        "Sorento": "джип/SUV", "Sportage": "джип/SUV", "K5": "седан", "Soul": "хэтчбек",
        # Mazda
        "Axela": "хэтчбек", "CX-5": "джип/SUV", "Demio": "хэтчбек", "Mazda3": "седан",
        "Mazda6": "седан", "Familia": "универсал", "MPV": "минивэн",
        # Ford
        "C-MAX": "минивэн", "Explorer": "джип/SUV", "Fiesta": "хэтчбек", "Focus": "седан",
        "Fusion": "седан", "Kuga": "джип/SUV", "Mondeo": "седан",
        # Renault
        "Arkana": "лифтбек", "Duster": "джип/SUV", "Kaptur": "джип/SUV", "Logan": "седан",
        "Megane": "хэтчбек", "Sandero": "хэтчбек", "Sandero Stepway": "хэтчбек",
        # Lexus
        "ES250": "седан", "GX460": "джип/SUV", "LX570": "джип/SUV", "NX200": "джип/SUV",
        "RX300": "джип/SUV", "RX350": "джип/SUV", "RX200t": "джип/SUV", "RX450h": "джип/SUV",
        # Skoda
        "Fabia": "хэтчбек", "Karoq": "джип/SUV", "Kodiaq": "джип/SUV", "Octavia": "лифтбек",
        "Rapid": "лифтбек", "Superb": "лифтбек", "Yeti": "джип/SUV",
        # Suzuki
        "Escudo": "джип/SUV", "Grand Vitara": "джип/SUV", "Jimny": "джип/SUV",
        "Jimny Sierra": "джип/SUV", "Solio": "минивэн", "Swift": "хэтчбек", "SX4": "хэтчбек",
        # Nissan
        "Note": "хэтчбек", "X-Trail": "джип/SUV", "Qashqai": "джип/SUV", "Almera": "седан",
        "Serena": "минивэн", "Juke": "хэтчбек", "Murano": "джип/SUV",
        # Geely
        "Okavango": "минивэн", "Monjaro": "джип/SUV", "Emgrand": "седан",
        "Atlas": "джип/SUV", "Atlas Pro": "джип/SUV", "Cityray": "седан", "Tugella": "джип/SUV",
        # Mercedes-Benz
        "S-Class": "седан", "GL-Class": "джип/SUV", "G-Class": "джип/SUV", "E-Class": "седан",
        "CLA-Class": "седан", "C-Class": "седан", "A-Class": "хэтчбек",
        # Opel
        "Antara": "джип/SUV", "Astra": "хэтчбек", "Astra GTC": "хэтчбек", "Corsa": "хэтчбек",
        "Insignia": "лифтбек", "Mokka": "джип/SUV", "Vectra": "седан", "Zafira": "минивэн",
        # Chery
        "Tiggo T11": "джип/SUV", "Tiggo 8 Pro Max": "джип/SUV", "Tiggo 7 Pro Max": "джип/SUV",
        "Tiggo 7 Pro": "джип/SUV", "Tiggo 4": "джип/SUV", "Arrizo 8": "седан", "Tiggo 4 Pro": "джип/SUV",
        # Haval
        "M6": "джип/SUV", "F7x": "джип/SUV", "Jolion": "джип/SUV", "H9": "джип/SUV",
        "H3": "джип/SUV", "F7": "джип/SUV", "Dargo": "джип/SUV",
        # Honda
        "Insight": "седан", "Vezel": "джип/SUV", "Stepwgn": "минивэн", "Fit": "хэтчбек",
        "CR-V": "джип/SUV", "Civic": "седан", "Accord": "седан",
        # Toyota
        "Corolla Fielder": "универсал", "Corolla": "седан", "Carina": "седан", "Camry": "седан",
        "Caldina": "универсал", "Avensis": "лифтбек", "Harrier": "джип/SUV",
        "Land Cruiser": "джип/SUV", "Mark II": "седан", "Land Cruiser Prado": "джип/SUV",
        "Passo": "хэтчбек", "Wish": "универсал",
        # Volkswagen
        "Tiguan": "джип/SUV", "Touareg": "джип/SUV", "Transporter": "минивэн", "Passat CC": "лифтбек",
        "Jetta": "седан", "Passat": "седан", "Golf": "хэтчбек",
    }

    # Заполняем body_type на основе model
    df['body_type'] = df.apply(
        lambda row: model_to_body_type.get(row['model'], row['body_type']) 
        if row['body_type'] == 'Нет данных' else row['body_type'],
        axis=1
    )

    # Удаляем дубликаты по строкам
    df = df.drop_duplicates()

    # Сбрасываем индексы после удаления дубликатов
    df = df.reset_index(drop=True)

    df.to_csv('cleaned.csv', index=False)

    df.drop(columns=['additional_info', 'image_url', 'listing_url', 'complectation'], inplace=True)

    return df

def preprocess_data(df, target_column):
    # One-Hot Encoding для столбца 'color'
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    color_encoded = pd.DataFrame(
        ohe.fit_transform(df[['color']]),
        columns=ohe.get_feature_names_out(['color'])
    )
    
    # Объединение закодированных столбцов с оригинальным DataFrame
    df = pd.concat([df.drop('color', axis=1), color_encoded], axis=1)

    # Target Encoding для остальных категориальных признаков
    target_encoder = TargetEncoder()
    categorical_columns = ['brand', 'model', 'location', 'type_engine', 'transmission', 'drive', 'body_type', 'steering']
    df[categorical_columns] = target_encoder.fit_transform(df[categorical_columns], df[target_column])

    # Определение числовых столбцов (кроме целевой переменной)
    numerical_columns = ['year', 'volume_engine', 'power', 'mileage']

    # Инициализация MinMaxScaler для нормализации числовых столбцов
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df


def process_data(df):
    # Загрузка данных (замените этот код на вашу логику загрузки данных)
    # Применяем последовательную обработку через функции
    df = process_car_data_1(df)
    df = process_car_data_2(df)
    df = process_car_data_3(df)
    df = process_car_data_4(df)
    df = preprocess_data(df, 'price')
    df.to_csv('transformed.csv', index=False)

    # Сохранение или дальнейшая работа с очищенными данными
    return df

# df = process_data(df)





dag = DAG(
    'car_data_processing',
    retry_delay = timedelta(minutes=5),
    owner= 'AutoChoice',
    description='Pipeline for processing car data',
    start_date=datetime(2024, 11, 19),
    schedule_interval='0 15 * * *',
    catchup=False,
)

# Определяем задачу обработки данных
process_task = PythonOperator(
    task_id='process_car_data',
    python_callable=process_data,
    provide_context=True,
    dag=dag,
)

process_task