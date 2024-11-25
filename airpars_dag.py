from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import time
import csv
import undetected_chromedriver.v2 as uc
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def parse_listing(url, attempts=5):
    for attempt in range(attempts):
        try:
            print(f'Попытка {attempt + 1}/{attempts} для {url}')
            # Ожидание до 2 минут для ответа от сервера
            start_time = time.time()
            while True:
                try:
                    driver.get(url)
                    # Ожидание, пока основной элемент загружается
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.css-1kb7l9z')))
                    break  # выходим из цикла, если страница загружена
                except Exception as e:
                    if time.time() - start_time > 120:  # Проверяем прошло ли больше 2 минут
                        print(f'Время ожидания превышено для {url}, пытаемся снова...')
                        raise  # Позволяем выйти из текущей попытки
                time.sleep(2)  # Не перегружайте сервер, ждем перед следующей попыткой

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Код для извлечения данных из soup остается без изменений
            title_main = soup.select_one('span.css-1kb7l9z').text.strip() if soup.select_one('span.css-1kb7l9z') else 'Нет данных'
            title_secondary = soup.select_one('h1.css-1soizd2').text.strip() if soup.select_one('h1.css-1soizd2') else 'Нет данных'
            title = f"{title_main} - {title_secondary}" if title_main != 'Нет данных' else title_secondary

            price_tag = soup.select_one('div.wb9m8q0')
            price = price_tag.text.strip() if price_tag else 'Нет данных'

            location = soup.select_one('span.css-17lk78h').text.strip() if soup.select_one('span.css-17lk78h') else 'Нет данных'

            description_tag = soup.select_one('div.css-1j2epfn')
            description = description_tag.get_text(separator=' ', strip=True) if description_tag else 'Нет данных'

            specifications = {}
            spec_tags = soup.select('tr')
            for spec in spec_tags:
                key = spec.select_one('th').text.strip() if spec.select_one('th') else 'Нет данных'
                value = spec.select_one('td').text.strip() if spec.select_one('td') else 'Нет данных'
                specifications[key] = value

            engine = specifications.get('Двигатель', 'Нет данных')
            power = specifications.get('Мощность', 'Нет данных')
            transmission = specifications.get('Коробка передач', 'Нет данных')
            drive = specifications.get('Привод', 'Нет данных')
            body_type = specifications.get('Тип кузова', 'Нет данных')
            color = specifications.get('Цвет', 'Нет данных')
            mileage = specifications.get('Пробег', 'Нет данных')
            steering = specifications.get('Руль', 'Нет данных')
            generation = specifications.get('Поколение', 'Нет данных')
            complectation = specifications.get('Комплектация', 'Нет данных')

            additional_info_tag = soup.select_one('div.css-inmjwf')
            additional_info = additional_info_tag.get_text(separator=' ', strip=True) if additional_info_tag else 'Нет данных'

            image_tag = soup.select_one('img.css-qy78xy')
            image_url = image_tag['src'] if image_tag and image_tag.get('src') else 'Нет данных'

            return {
                'title': title,
                'price': price,
                'location': location,
                'description': description,
                'engine': engine,
                'power': power,
                'transmission': transmission,
                'drive': drive,
                'body_type': body_type,
                'color': color,
                'mileage': mileage,
                'steering': steering,
                'generation': generation,
                'complectation': complectation,
                'additional_info': additional_info,
                'image_url': image_url,
                'listing_url': url
            }
        except Exception as e:
            print(f'Ошибка при парсинге данных объявления: {e}. Попытка {attempt + 1}/{attempts}')
            time.sleep(5)  # Задержка перед очередной попыткой

    return None  # Возвращает None, если не удалось получить данные после всех попыток



def get_listing_links(page_number, brand, model):
    base_url = f'https://auto.drom.ru/{brand}/{model}/page{page_number}/'
    driver.get(base_url)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser') 
    links = [
        item.find('a', {'data-ftid': 'bull_title'})['href']
        for item in soup.select("div[data-ftid='bulls-list_bull']")
        if item.find('a', {'data-ftid': 'bull_title'}) and item.find('a', {'data-ftid': 'bull_title'})['href']
    ]
    
    if not links:
        print(f'Объявления отсутствуют на странице {page_number} для модели {model} бренда {brand}')
    
    return links

def scrape_cars_drom():
    output_dir = '/home/aleksey/DS_bootcamp/ФИНПРОЕКТ'
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, 'airflow.csv')

    csv_columns = [
        'title', 'price', 'location', 'description', 'engine', 'power',
        'transmission', 'drive', 'body_type', 'color', 'mileage', 'steering',
        'generation', 'complectation', 'additional_info', 'image_url', 'listing_url'
    ]
    
    driver = uc.Chrome(options=uc.ChromeOptions(), version_main=130)
    
    car_models = {
    'kia': ['cee~d', 'cerato', 'optima', 'rio', 'sorento', 'sportage', 'k5', 'soul'],
    'mazda': ['axela', 'cx-5', 'demio', 'mazda3', 'mazda6', 'familia', 'mpv'],
    'ford': ['c-max', 'explorer', 'fiesta', 'focus', 'fusion', 'kuga', 'mondeo'],
    'renault': ['arkana', 'duster', 'kaptur', 'logan', 'megane', 'sandero', 'sandero_stepway'],
    'lexus': ['es250', 'gx460', 'lx570', 'nx200', 'rx300', 'rx350', 'rx200t', 'rx450h'],
    'skoda': ['fabia', 'karoq', 'kodiaq', 'octavia', 'rapid', 'superb', 'yeti'],
    'suzuki': ['escudo', 'grand_vitara', 'jimny', 'jimny_sierra', 'solio', 'swift', 'sx4'],
    'nissan': ['note', 'x-trail', 'qashqai', 'almera', 'serena', 'juke', 'murano'],
    'audi': ['a3', 'a4', 'a5', 'a6', 'q3', 'q5', 'q7'],
    'bmw': ['5-series', '3-series', '7-series', 'x5', 'x6', 'x3', 'x1'],
    'changan': ['cs35_plus', 'cs55_plus', 'uni-k', 'uni-s', 'uni-t', 'uni-v', 'alsvin'],
    'chevrolet': ['aveo', 'captiva', 'cruze', 'lacetti', 'lanos', 'niva', 'spark'],
    'hyundai': ['accent', 'solaris', 'getz', 'sonata', 'tucson', 'creta', 'elantra', 'santa_fe'],
    'mitsubishi': ['lancer', 'pajero_sport', 'rvr', 'galant', 'delica', 'asx', 'outlander'],
    'subaru': ['forester', 'impreza', 'legacy', 'legacy_b4', 'levorg', 'outback', 'xv'],
    'lada': ['granta', 'priora', 'vesta', '2114', '2115', 'largus', '2121_4x4_niva'],
    'geely': ['okavango', 'monjaro', 'emgrand_iv', 'atlas', 'atlas_pro', 'cityray', 'tugella'],
    'mercedes-benz': ['s-class', 'gl-class', 'g-class', 'e-class', 'cla-class', 'c-class', 'a-class'],
    'opel': ['antara', 'astra', 'astra_gtc', 'corsa', 'insignia', 'mokka', 'vectra', 'zafira'],
    'chery': ['tiggo', 'tiggo_8_pro_max', 'tiggo_7_pro_max', 'tiggo_7_pro', 'tiggo_4', 'arrizo_8', 'tiggo_4_pro'],
    'haval': ['m6', 'f7x', 'jolion', 'h9', 'h3', 'f7', 'dargo'],
    'honda': ['insight', 'vezel', 'stepwgn', 'fit', 'CR-V', 'civic', 'accord'],
    'toyota': ['corolla_fielder', 'corolla', 'carina', 'camry', 'caldina', 'avensis', 'harrier', 
               'land_cruiser', 'mark_ii', 'land_cruiser_prado', 'passo', 'wish'],
    'volkswagen': ['tiguan', 'touareg', 'transporter', 'passat_cc', 'jetta', 'passat', 'golf']
}

    all_data = []
    max_pages = 1


    for brand, models in car_models.items():
        for model in models:
            for page in range(1, max_pages + 1):
                print(f'Парсинг страницы {page} для модели {model} бренда {brand}')
                links = get_listing_links(page, brand, model)
                
                if not links:
                    print(f'Достигнут конец списка для модели {model} бренда {brand} на странице {page}')
                    break
                
                for link in links:
                    print(f'  Парсинг объявления {link}')
                    data = parse_listing(link)
                    if data:
                        all_data.append(data)
                    time.sleep(1)

    # Сохранение данных в CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_data)

    print(f'Данные успешно сохранены в {csv_file_path}')
    driver.quit()

# Определение DAG
default_args = {
    'owner': 'AutoChoice',
    'start_date': datetime(2024, 11, 18),
    'retries': 1,
}

dag = DAG(
    'car_scraper_dag',
    default_args=default_args,
    retry_delay = timedelta(minutes=5),
    description='DAG для парсинга объявлений автомобилей.',
    schedule_interval='0 10 * * *',  # Каждый день в 10:00
)

scrape_task = PythonOperator(
    task_id='scrape_cars',
    python_callable=scrape_cars_drom,
    dag=dag,
)

scrape_task