import os
import re
from flask import Flask, render_template, request
import json
import nltk
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
########################################################################################################################
##################################################*PATHS*###############################################################
BASE_DICTS = 'Convert_dicts'
BASE_DATA_PATH = 'sorted_data_acl'
basedir_model_save = 'LSTM_bin'
BEST_WEIGHTS_STORE_DIR = 'Best_weights'

########################################################################################################################
################################################*FUNCTIONS*#############################################################
def clean_text(text, chars_to_ignore_regex=CHARS_TO_IGNORE_REGEX):
    # Убираем игнорируемые символы
    cleaned_text = re.sub(chars_to_ignore_regex, '', text)
    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
#-----------------------------------------------------------------------------------------------------------------------
def tokenize_and_pad(text, word2index, max_len=MAX_SENTENCE_LENGTH):
    words = nltk.word_tokenize(text)
    seq = [word2index.get(word, word2index["UNK"]) for word in words]
    padded_seq = pad_sequences([seq], maxlen=max_len, padding='post')
    return padded_seq
#-----------------------------------------------------------------------------------------------------------------------
# Функция для загрузки словаря из JSON
def load_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



app = Flask(__name__)
# Загрузка модели и словаря
"""
global graph
graph = tf.compat.v1.get_default_graph()
session = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(session)
"""

# Загрузка модели и словаря
#model = load_model(os.path.join(BEST_WEIGHTS_STORE_DIR, 'LSTM_dist_best_weihts.bin'))

#with open('Convert_dicts/word2index.json', 'r') as f:
#    word2index = json.load(f)
# Загрузка словарей
word2index = load_dict_from_json(os.path.join(BASE_DICTS,'word2index.json'))
index2word = load_dict_from_json(os.path.join(BASE_DICTS,'index2word.json'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    review_text = ""
    if request.method == 'POST':
        review_text = request.form['review_text']
        cleaned_text = clean_text(review_text)
        # Усечение до нужной длины
        truncated_text = cleaned_text[:MAX_CHAR_LENGTH]
        tokenized_text = tokenize_and_pad(truncated_text, word2index)

        # Загружаем модель и создаем граф и сессию для каждого предсказания
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session(graph=graph)
            with session.as_default():
                model = load_model(os.path.join(BEST_WEIGHTS_STORE_DIR, 'LSTM_dist_best_weihts.bin'))
                prediction_prob = model.predict(tokenized_text)[0][0]
                prediction = "Positive" if round(prediction_prob) == 1 else "Negative"

            # Закрываем сессию, чтобы освободить ресурсы GPU
            session.close()

    return render_template('index.html', prediction=prediction, review_text=review_text)


if __name__ == '__main__':
    app.run(debug=True)
