from django.shortcuts import render
from .ml.train import predict_products, get_website_text
from transformers import BertForTokenClassification, PreTrainedTokenizerFast
import os


def index(request):
    found_products = []
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ml/custom_ner_model')

    if request.method == 'POST':
        url = request.POST.get('url', '')

        try:
            # Загрузка предварительно обученной модели
            model = BertForTokenClassification.from_pretrained(MODEL_DIR)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)

            # Получаем текст с сайта
            website_text = get_website_text(url)

            if website_text:
                found_products = predict_products(website_text, model, tokenizer)

            if not found_products:
                found_products = [f"Товар {i}" for i in range(1, 4)]

        except Exception as e:
            found_products = [f"Ошибка: {str(e)}"]

    return render(request, 'myapp/index.html', {'products': found_products})