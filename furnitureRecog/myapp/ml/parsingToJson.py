import csv
import json
import requests
from bs4 import BeautifulSoup
import chardet
import time

# Настройки
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TIMEOUT = 15
MAX_RETRIES = 2
DELAY_BETWEEN_REQUESTS = 1  # секунда
MAX_URLS_TO_PROCESS = 1000  # Ограничение на количество обрабатываемых URL


def detect_encoding(file_path):
    """Определяет кодировку файла"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def fetch_html(url):
    """Получает HTML страницы с повторными попытками и задержкой"""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                allow_redirects=True,
                verify=True
            )

            if response.status_code != 200:
                print(f"Страница {url} вернула статус {response.status_code}")
                continue

            if response.encoding is None or response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'

            return response.text

        except requests.exceptions.RequestException as e:
            print(f"Попытка {attempt + 1} для {url} не удалась: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2)

    return None


def process_row(row):
    """Обрабатывает одну строку из CSV"""
    try:
        url = row['max(page)']
        products = row['result']

        # Пропускаем записи с пустым result
        if not products or products.strip() == '':
            return None

        html = fetch_html(url)
        if html is None:
            print(f"Не удалось загрузить страницу после {MAX_RETRIES} попыток: {url}")
            html = f"Не удалось загрузить страницу: {url}"
        else:
            soup = BeautifulSoup(html, 'html.parser')
            html = soup.get_text(separator='\n', strip=True)

        return {
            "text": html,
            "result": "ERROR" if products == 'ERROR' else [item.strip() for item in products.split('%%') if
                                                           item.strip()],
            "url": url
        }
    except Exception as e:
        print(f"Ошибка обработки строки {row.get('max(page)', 'N/A')}: {str(e)}")
        return None


def csv_to_json(csv_file_path, output_json_path):
    """Основная функция обработки CSV и сохранения в JSON"""
    try:
        encoding = detect_encoding(csv_file_path)
        print(f"Определена кодировка файла: {encoding}")
    except Exception as e:
        print(f"Ошибка определения кодировки: {str(e)}")
        encoding = 'utf-8'

    result_data = []
    processed_urls = set()
    urls_with_empty_result = 0

    try:
        with open(csv_file_path, mode='r', encoding=encoding) as file:
            try:
                csv_reader = csv.DictReader(file, delimiter=';')
                fieldnames = csv_reader.fieldnames
            except:
                file.seek(0)
                csv_reader = csv.DictReader(file, delimiter=',')
                fieldnames = csv_reader.fieldnames

            print(f"Заголовки столбцов: {fieldnames}")

            # Обрабатываем строки до тех пор, пока не найдем непустой result
            for row in csv_reader:
                url = row['max(page)']
                products = row['result']

                if url in processed_urls:
                    continue

                processed_urls.add(url)

                if not products or products.strip() == '':
                    urls_with_empty_result += 1
                    continue

                # Ограничение на количество обрабатываемых URL
                if len(result_data) >= MAX_URLS_TO_PROCESS:
                    print(f"Достигнуто максимальное количество URL для обработки ({MAX_URLS_TO_PROCESS})")
                    break

                result = process_row(row)
                if result is not None:
                    result_data.append(result)

    except Exception as e:
        print(f"Ошибка при чтении CSV: {str(e)}")
        return None

    print(f"\nНайдено URL с пустым result: {urls_with_empty_result}")
    print(f"Успешно обработано URL: {len(result_data)}")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\nУспешно сохранено {len(result_data)} записей в {output_json_path}")
    except Exception as e:
        print(f"Ошибка при сохранении JSON: {str(e)}")
        return None

    return result_data


if __name__ == "__main__":
    csv_path = "templates/myapp/URL_list_mark.csv"
    json_path = "output.json"

    try:
        data = csv_to_json(csv_path, json_path)

        if data:
            success_count = sum(1 for entry in data if not entry['text'].startswith("Не удалось"))
            error_count = len(data) - success_count

            print("\nРезультаты обработки:")
            print(f"Всего записей: {len(data)}")
            print(f"Успешно загружено: {success_count}")
            print(f"Не удалось загрузить: {error_count}")

            print("\nПримеры успешных запросов:")
            for entry in [e for e in data if not e['text'].startswith("Не удалось")][:2]:
                print(f"\nURL: {entry['url']}")
                print(f"HTML (начало): {entry['text'][:100]}...")

            print("\nПримеры неудачных запросов:")
            for entry in [e for e in data if e['text'].startswith("Не удалось")][:2]:
                print(f"\nURL: {entry['url']}")
                print(f"Причина: {entry['text']}")

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")