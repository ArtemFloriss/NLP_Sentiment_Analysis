import os
import random
import re
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models
from lxml import etree
import json

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
best_weights_store_dir = 'Best_weights'

########################################################################################################################
################################################*FUNCTIONS*#############################################################
def clean_text(text, chars_to_ignore_regex):
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
def clean_xml_file(input_file, output_file):
    # Открываем исходный файл и читаем его содержимое
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Удаляем недопустимые символы
    # Разрешенные символы: [\x09\x0A\x0D\x20-\xD7FF\xE000-\xFFFD\x10000-\x10FFFF]
    cleaned_content = re.sub(r'[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]', '', content)

    # Записываем очищенный контент в новый файл
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

#-----------------------------------------------------------------------------------------------------------------------
def find_mismatched_tags(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Поиск открывающих и закрывающих тегов
    opening_tags = re.findall(r'<([a-zA-Z_][a-zA-Z0-9_\-]*)\b[^>]*>', content)
    closing_tags = re.findall(r'</([a-zA-Z_][a-zA-Z0-9_\-]*)\b[^>]*>', content)

    # Проверка на несоответствие количества открывающих и закрывающих тегов
    if len(opening_tags) != len(closing_tags):
        print(f"Mismatch in number of tags: {len(opening_tags)} opening tags and {len(closing_tags)} closing tags.")
        return

    # Сравнение открывающих и закрывающих тегов
    for i, tag in enumerate(opening_tags):
        if i >= len(closing_tags):
            print(f"Extra opening tag found: <{tag}> at position {i + 1}")
            continue

        if tag != closing_tags[i]:
            print(f"Mismatch at position {i + 1}: <{tag}> does not match </{closing_tags[i]}>")
            continue

    # Проверка на лишние закрывающие теги
    if len(closing_tags) > len(opening_tags):
        print(f"Extra closing tags found starting from position {len(opening_tags) + 1}")
#-----------------------------------------------------------------------------------------------------------------------
def fix_mismatched_tags(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Ищем теги с именами reviewer_location и span и исправляем их
    content = re.sub(r'</span>\s*</reviewer_location>', '</reviewer_location>', content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
#-----------------------------------------------------------------------------------------------------------------------
# Функция для предсказания
def predict_review(xml_file, model, chars_to_ignore_regex = CHARS_TO_IGNORE_REGEX, max_char_length = MAX_CHAR_LENGTH, max_sent_length = MAX_SENTENCE_LENGTH):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Выбор случайного отзыва
    reviews = root.findall('review')
    chosen_review = random.choice(reviews)
    review_text = chosen_review.find('review_text').text.strip()

    # Очистка текста
    cleaned_text = clean_text(review_text, chars_to_ignore_regex)

    # Усечение до нужной длины
    truncated_text = cleaned_text[:max_char_length]
    print(truncated_text)

    # Токенизация и преобразование в последовательность индексов
    words = nltk.word_tokenize(truncated_text.lower())
    seqs = [word2index.get(word, word2index['UNK']) for word in words]

    # Паддинг
    X = sequence.pad_sequences([seqs], maxlen=max_sent_length)

    # Предсказание
    prediction = model.predict(X)
    prediction = round(prediction[0][0])

    # Вывод результата
    if prediction == 1:
        print("Отзыв положительный.")
    else:
        print("Отзыв отрицательный.")


########################################################################################################################
# Загрузка модели
model_path = os.path.join(best_weights_store_dir, 'LSTM_dist_best_weihts.bin')
model = models.load_model(model_path)

# Загрузка словарей
word2index = load_dict_from_json(os.path.join(BASE_DICTS,'word2index.json'))
index2word = load_dict_from_json(os.path.join(BASE_DICTS,'index2word.json'))

# Путь к файлу 'unlabeled_format_all.xml'
input_xml_file = os.path.join(BASE_DATA_PATH, 'unlabeled_format_all_small.xml')
output_xml_file = os.path.join(BASE_DATA_PATH, 'unlabeled_format_all_cleaned.xml')
# Чистим XML файл
clean_xml_file(input_xml_file, output_xml_file)
# Поиск несовпадающих тегов
find_mismatched_tags(input_xml_file)
# Путь к файлу 'unlabeled_format_all_cleaned.xml'
unlabeled_file = os.path.join(BASE_DATA_PATH, 'unlabeled_format_all_cleaned.xml')

# Выполнение предсказания
predict_review(unlabeled_file, model=model, chars_to_ignore_regex=CHARS_TO_IGNORE_REGEX)
