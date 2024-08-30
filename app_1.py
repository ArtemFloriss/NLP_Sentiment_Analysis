from flask import Flask, request, render_template
import json
import requests
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import nltk
import re
#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

MAX_FEATURES = 10000
MAX_CHAR_LENGTH = 1000
MAX_SENTENCE_LENGTH = 200

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 12

CHARS_TO_IGNORE_REGEX = r'[^a-zA-Z0-9\s]'
##################################################*PATHS*###############################################################
BASE_DICTS = 'Convert_dicts'
BASE_DATA_PATH = 'sorted_data_acl'
basedir_model_save = 'LSTM_bin'
BEST_WEIGHTS_STORE_DIR = 'Best_weights'

################################################*FUNCTIONS*#############################################################
def clean_text(text, chars_to_ignore_regex=CHARS_TO_IGNORE_REGEX):
    # Убираем игнорируемые символы
    cleaned_text = re.sub(chars_to_ignore_regex, '', text)
    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
#-----------------------------------------------------------------------------------------------------------------------
# Функция для загрузки словаря из JSON
def load_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
#-----------------------------------------------------------------------------------------------------------------------
def tokenize_and_pad(text, word2index, max_len=MAX_SENTENCE_LENGTH):
    words = nltk.word_tokenize(text)
    seq = [word2index.get(word, word2index["UNK"]) for word in words]
    padded_seq = pad_sequences([seq], maxlen=max_len, padding='post')
    return padded_seq
#-----------------------------------------------------------------------------------------------------------------------

app = Flask(__name__)

# URL TensorFlow Serving для модели
TF_SERVING_URL = 'http://localhost:8501/v1/models/sentiment_model:predict'


def preprocess_input(text):
    # Загрузка словарей
    word2index = load_dict_from_json(os.path.join(BASE_DICTS, 'word2index.json'))
    index2word = load_dict_from_json(os.path.join(BASE_DICTS, 'index2word.json'))

    tokenized_text = tokenize_and_pad(text, word2index)

    return tokenized_text


def predict_sentiment(text):
    # Предобработка текста
    tokenized_text = preprocess_input(text)

    # Подготовка данных для отправки на TensorFlow Serving
    data = json.dumps({"instances": tokenized_text.tolist()})

    # Запрос на инференс
    response = requests.post(TF_SERVING_URL, data=data)

    # Проверка на успешный ответ
    if response.status_code != 200:
        raise ValueError(f"Error: received response {response.status_code}, {response.text}")

    result = response.json()

    # Обработка возможных ключей в ответе
    if 'predictions' in result:
        prediction = result['predictions'][0][0]
    elif 'outputs' in result:
        prediction = result['outputs'][0][0]
    else:
        raise ValueError(f"Unexpected response format: {result}")

    return prediction


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        review_text = request.form['review_text']
        cleaned_text = clean_text(review_text)
        # Усечение до нужной длины
        truncated_text = cleaned_text[:MAX_CHAR_LENGTH]

        sentiment = predict_sentiment(truncated_text)
        sentiment_label = "Positive" if round(sentiment) == 1 else "Negative"

    return render_template('index.html', prediction=sentiment_label, review_text=review_text)
    #return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
