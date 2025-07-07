import json
import string
import requests
from bs4 import BeautifulSoup
from transformers import BertConfig, BertForTokenClassification
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
import torch
import numpy as np
import asyncio
from playwright.async_api import async_playwright

# 1. Инициализация устройства
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Функция для загрузки текста с сайта
async def async_get_website_text(url):
    """Асинхронная версия функции для получения текста с сайта"""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Настройки для лучшего парсинга
            await page.set_viewport_size({"width": 1280, "height": 720})
            await page.goto(url, wait_until="networkidle", timeout=90000)

            # Ожидание дополнительных элементов при необходимости
            try:
                await page.wait_for_load_state("networkidle", timeout=30000)
            except:
                pass

            content = await page.content()
            await browser.close()

            soup = BeautifulSoup(content, 'html.parser')

            # Удаление ненужных элементов
            for element in soup(['script', 'style', 'nav', 'footer',
                               'iframe', 'svg', 'img', 'button']):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)
            return text

    except Exception as e:
        print(f"Ошибка при загрузке сайта: {e}")
        return None

def get_website_text(url):
    """Синхронная обертка для асинхронной функции"""
    try:
        # Проверяем, есть ли запущенный event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        # Если loop запущен, используем create_task
        if loop and loop.is_running():
            async def wrapper():
                return await async_get_website_text(url)
            future = asyncio.ensure_future(wrapper())
            return loop.run_until_complete(future)
        else:
            # Иначе создаем новый loop
            return asyncio.run(async_get_website_text(url))
    except Exception as e:
        print(f"Ошибка в синхронной обертке: {e}")
        return None

# 3. Загрузка и подготовка данных
def load_and_prepare_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем формат данных
        if not isinstance(data, list):
            print("Ошибка: Данные должны быть массивом объектов")
            return None

        # Фильтруем невалидные записи
        valid_data = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "text" not in item or "result" not in item:
                continue
            if not isinstance(item["text"], str) or not isinstance(item["result"], list):
                continue

            # Фильтруем невалидные названия товаров
            valid_results = [res for res in item["result"] if isinstance(res, str) and res.strip()]
            if not valid_results:
                continue

            valid_data.append({
                "text": item["text"],
                "result": valid_results
            })

        if not valid_data:
            print("Ошибка: Нет валидных данных после проверки")
            return None

        return valid_data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def create_tags_from_result(example):
    text = example["text"]
    result_products = example["result"]

    # Разбиваем текст на строки (разделитель \n)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Инициализируем теги (0 для каждой строки)
    tags = [0] * len(lines)

    # Очищаем названия товаров для сравнения
    cleaned_products = []
    for product in result_products:
        # Удаляем пунктуацию и приводим к нижнему регистру
        cleaned = product.strip().strip(string.punctuation).lower()
        if cleaned:
            cleaned_products.append(cleaned)

    # Помечаем строки, содержащие товары
    for i, line in enumerate(lines):
        # Очищаем строку для сравнения
        cleaned_line = line.strip().strip(string.punctuation).lower()

        # Проверяем, содержит ли строка любой из товаров
        for product in cleaned_products:
            if product in cleaned_line:
                tags[i] = 1
                break  # Помечаем строку как товар и переходим к следующей

    return {
        "text": '\n'.join(lines),  # Возвращаем текст с очищенными строками
        "tags": tags
    }

# 4. Создание и обучение модели
def train_model(train_data):
    if not train_data:
        print("Ошибка: Нет данных для обучения")
        return None, None

    # Создаем словарь
    vocab = set()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "\n"]

    for item in train_data:
        for word in item["text"].split():
            vocab.add(word)

    vocab = {word: idx for idx, word in enumerate(special_tokens + sorted(vocab))}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # Подготовка данных
    formatted_data = []
    for example in train_data:
        tagged = create_tags_from_result(example)
        if tagged:
            formatted_data.append(tagged)

    def prepare_dataset(data, tokenizer, max_length=256):
        input_ids = []
        attention_masks = []
        labels = []

        for item in data:
            tokens = item["text"].split()
            tags = item["tags"]

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            word_ids = encoding.word_ids()
            aligned_labels = []

            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    aligned_labels.append(-100)
                else:
                    if word_id >= len(tokens):
                        aligned_labels.append(-100)
                    elif tokens[word_id] == '\n':
                        aligned_labels.append(-100)
                    else:
                        aligned_labels.append(tags[word_id] if word_id < len(tags) else -100)

            aligned_labels += [-100] * (max_length - len(aligned_labels))

            input_ids.append(encoding['input_ids'].squeeze())
            attention_masks.append(encoding['attention_mask'].squeeze())
            labels.append(torch.tensor(aligned_labels))

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels)
        }

    processed_data = prepare_dataset(formatted_data, tokenizer)
    dataset = Dataset.from_dict({
        'input_ids': processed_data['input_ids'].numpy(),
        'attention_mask': processed_data['attention_mask'].numpy(),
        'labels': processed_data['labels'].numpy()
    })

    # Создаем модель
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        num_labels=2
    )
    model = BertForTokenClassification(config).to(device)

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir="./custom_ner_model",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        save_steps=200,
        logging_steps=10,
        report_to="none",
        disable_tqdm=False,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        no_cuda=True if device.type == 'mps' else False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Обучение
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nНачало обучения модели...")
    try:
        trainer.train()
        print("Обучение завершено!")
        return model, tokenizer
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return None, None

# 5. Функция для предсказания товаров
def predict_products(text, model, tokenizer, max_length=256):
    if not text or not model or not tokenizer:
        print("Ошибка: Неверные входные данные для предсказания")
        return []

    model.to(device)
    model.eval()

    tokens = text.split()
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    word_ids = inputs.word_ids()
    current_product = []
    products = []

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id >= len(tokens):
            continue

        token = tokens[word_id]

        if predictions[i] == 1:
            current_product.append(token)
        elif current_product:
            products.append(" ".join(current_product))
            current_product = []

    if current_product:
        products.append(" ".join(current_product))

    return products

# Основной поток выполнения
if __name__ == "__main__":
    # Загрузка данных
    print("Загрузка данных...")
    train_data = load_and_prepare_data("furnitureRecog/furnitureRecog/myapp/output.json")
    if not train_data:
        exit()

    # Обучение модели
    print("Подготовка к обучению...")
    model, tokenizer = train_model(train_data)
    if not model or not tokenizer:
        exit()

    # Тестирование
    while True:
        print("\n1. Протестировать на URL сайта")
        print("2. Протестировать на введенном тексте")
        print("3. Выход")
        choice = input("Выберите вариант: ")

        if choice == '1':
            website_url = input("Введите URL сайта: ")
            print(f"Анализ сайта {website_url}...")
            website_text = get_website_text(website_url)

            if website_text:
                found_products = predict_products(website_text, model, tokenizer)

                if found_products:
                    print("\nНайденные товары:")
                    for i, product in enumerate(found_products, 1):
                        print(f"{i}. {product}")
                else:
                    print("Товары не найдены")
            else:
                print("Не удалось получить текст с сайта")

        elif choice == '2':
            text = input("Введите текст для анализа: ")
            found_products = predict_products(text, model, tokenizer)

            if found_products:
                print("\nНайденные товары:")
                for i, product in enumerate(found_products, 1):
                    print(f"{i}. {product}")
            else:
                print("Товары не найдены")

        elif choice == '3':
            break

        else:
            print("Неверный выбор, попробуйте снова")

    # Сохранение модели
    model.save_pretrained("./custom_ner_model")
    tokenizer.save_pretrained("./custom_ner_model")
    print("\nМодель сохранена в папке 'custom_ner_model'")