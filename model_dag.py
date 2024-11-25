import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import subprocess
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def train_and_evaluate_model():
    # Загрузка данных
    df = pd.read_csv('/home/aleksey/DS_bootcamp/ФИНПРОЕКТ/transformed.csv')
    
    best_params = {'depth': 10, 'learning_rate': 0.14283458648427383, 'iterations': 1000,
    'l2_leaf_reg':   6.737889033032106}

    # Загружаем модель
    model = CatBoostRegressor(**best_params)
    # loaded_model.load_model('catboost_model.cbm')

    # Определение признаков и целевой переменной
    X = df.drop('price', axis=1)
    y = df['price']

    # Разделение данных
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Переобучение модели на новых данных
    model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = model.predict(X_valid)

    # Расчет MAE
    mae = mean_absolute_error(y_valid, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Сохранение модели
    model.save_model('catboost_model.cbm')

    # Запись MAE в текстовый файл
    with open('mae_results.txt', 'a') as f:  # Открытие файла в режиме добавления
        f.write(f"Mean Absolute Error: {mae}\n")  # Запись MAE в файл

    # Автоматическая отправка изменений на GitHub
    # Настройка команды git
    try:
        subprocess.run(['git', 'add', '.'], check=True)  # Добавляем изменения
        subprocess.run(['git', 'commit', '-m', f'Update model and MAE: {mae}'], check=True)  # Фиксируем изменения
        subprocess.run(['git', 'push'], check=True)  # Отправляем на удалённый репозиторий
    except subprocess.CalledProcessError as e:
        print(f'Error while pushing to GitHub: {e}')

default_args = {
    'owner': 'AutoChoice',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 19),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('model_training_dag',
         default_args=default_args,
         schedule_interval='0 16 * * *',  
         catchup=False) as dag:

    # Определение задачи
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_and_evaluate_model
    )
