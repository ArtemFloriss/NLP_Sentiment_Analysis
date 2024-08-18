# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation, Dense, Dropout, SpatialDropout1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
import matplotlib.pyplot as plt
import nltk
from tensorflow_core.python.keras.layers.pooling import GlobalAveragePooling1D

nltk.download('punkt')

import numpy as np
import os
from collections import defaultdict

import pandas as pd
import re
from lxml import etree
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from collections import OrderedDict

from cuml.svm import SVC as cuml_SVC

import joblib
import json
#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

MAX_FEATURES = 10000
MAX_SENTENCE_LENGTH = 200

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 12
########################################################################################################################
##################################################*PATHS*###############################################################
BASE_DICTS = 'Convert_dicts'
plot_out_dir = 'Output_plot'
basedir_model_save = 'LSTM_bin'
best_weights_store_dir = 'Best_weights'

########################################################################################################################
################################################*FUNCTIONS*#############################################################
def step_decay(epoch):
# initialize the base initial learning rate, drop factor, and
# epochs to drop every
    initAlpha = 0.001
    factor = 0.2
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    # return the learning rate
    return float(alpha)
#-----------------------------------------------------------------------------------------------------------------------
def save_dict_to_json(dict_obj, file_path): #Сохранение словарей кодирования/декодирования текста в json-файл
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict_obj, json_file, ensure_ascii=False, indent=4)
#-----------------------------------------------------------------------------------------------------------------------
def clean_text(text, chars_to_ignore_regex):
    # Убираем игнорируемые символы
    cleaned_text = re.sub(chars_to_ignore_regex, '', text)
    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
#-----------------------------------------------------------------------------------------------------------------------

def parse_emotion_file(xml_file, chars_to_ignore_regex):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    data = []

    for review in root.findall('review'):
        review_text = review.find('review_text').text.strip()
        is_positive = review.find('is_positive').text.strip()

        # Очищаем текст от спецсимволов и неинформационных символов
        review_text = clean_text(review_text, chars_to_ignore_regex)

        data.append({'review_text': review_text, 'is_positive': is_positive})

    return pd.DataFrame(data)
########################################################################################################################

# Путь к файлу 'emotion_format_all.xml'
emotion_file = os.path.join(ROOT_DIR, 'emotion_format_all.xml')

# Новое регулярное выражение для игнорирования всех спецсимволов и непечатных символов
chars_to_ignore_regex = r'[^a-zA-Z0-9\s]'

# Парсим XML и создаем DataFrame
df = parse_emotion_file(emotion_file, chars_to_ignore_regex)

# Выводим DataFrame для проверки
print(df.head())

# Опционально: Сохраняем DataFrame в файл CSV
df.to_csv('emotion_data.csv', index=False)

# Рассчитываем длины текстов отзывов
df['review_length'] = df['review_text'].apply(len)

# Вычисляем среднее значение, дисперсию и стандартное отклонение (СКО)
mean_length = df['review_length'].mean()
median_length = df['review_length'].median()
variance = df['review_length'].var()
std_dev = np.sqrt(variance)

# Определяем границы диапазона
lower_bound = median_length - std_dev
upper_bound = median_length + std_dev

upper_bound = min(upper_bound, 1000)
# Фильтруем строки, где длина отзыва находится в пределах [СРЗНАЧ - СКО; СРЗНАЧ + СКО]
filtered_df = df[(df['review_length'] >= lower_bound) & (df['review_length'] <= upper_bound)]

# Выводим среднее значение, дисперсию и СКО
print("Среднее значение длины отзыва:", mean_length)
print("Медианное значение длины отзыва:", median_length)
print("Дисперсия длины отзыва:", variance)
print("Среднеквадратическое отклонение (СКО):", std_dev)
# Строим гистограмму распределения длин текстов отзывов
plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['review_length'], kde=True, bins=20)
plt.title('Распределение длин текстов отзывов')
plt.xlabel('Длина текста отзыва')
plt.ylabel('Количество отзывов')
plt.show()

# Инициализируем необходимые переменные
word_freqs = defaultdict(int)
maxlen = 0
num_recs = 0

word_counts = Counter()
# Проходим по каждому отзыву в колонке 'review_text'
for review in filtered_df['review_text']:
    words = nltk.word_tokenize(review.lower())  # Токенизация и перевод в нижний регистр

    if len(words) > maxlen:
        maxlen = len(words)  # Обновляем максимальную длину
    """
    for word in words:
        word_freqs[word] += 1  # Обновляем частоту слов
    """
    num_recs += 1

    word_counts.update(words)

# Подсчитываем число уникальных токенов (размер словаря)
#vocab_size = len(word_freqs)
# Преобразуем словарь в список кортежей и сортируем по количеству вхождений
sorted_word_counts = word_counts.most_common()

# Выводим результаты
print(f"Максимальная длина отзыва (в токенах): {maxlen}")
print(f"Число уникальных токенов (размер словаря): {len(word_counts)}")

# Отфильтровываем слова, которые встречаются только один раз
filtered_vocab = {word: count for word, count in word_counts.items() if count > 2}
# Сортируем словарь по значениям (частотам) в порядке убывания
sorted_filtered_vocab = OrderedDict(sorted(filtered_vocab.items(), key=lambda item: item[1], reverse=True))
# Выбираем только первые MAX_FEATURES слов из отсортированного словаря
limited_vocab = dict(list(sorted_filtered_vocab.items())[:MAX_FEATURES])
"""
# Создаем две отдельные переменные для слов и их частот
words, counts = zip(*sorted_word_counts)

# Строим гистограмму
plt.figure(figsize=(14, 7))
plt.bar(words, counts)
plt.xticks(rotation=90)
plt.xlabel('Слова')
plt.ylabel('Частота вхождений')
plt.title('Распределение частот слов в текстах отзывов')
plt.show()
"""
VOCAB_SIZE = len(limited_vocab) + 2

# Сначала добавляем специальные токены
word2index = {"PAD": 0, "UNK": 1}

# Затем добавляем остальные слова
word2index.update({word: i+2 for i, (word, _) in enumerate(limited_vocab.items())})

index2word = {v:k for k, v in word2index.items()}

# Проверка
print("Размер словаря:", len(word2index))
print("Примеры из словаря word2index:", list(word2index.items())[:10])

# Сохраняем полученные словари в JSON-файлы
save_dict_to_json(word2index, os.path.join(BASE_DICTS,'word2index.json'))
save_dict_to_json(index2word, os.path.join(BASE_DICTS,'index2word.json'))

#Создаем датсет для обучения
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))

i = 0
for review, label in zip(filtered_df['review_text'], filtered_df['is_positive']):
    words = nltk.word_tokenize(review.lower())  # Токенизация и перевод в нижний регистр

    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])

    X[i] = seqs
    y[i] = int(label)
    i += 1

X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# Проверка результатов
print("Пример закодированной последовательности:", X[0])
print("Метка для первого примера:", y[0])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


#---------------------------------------------*Model Initialize*--------------------------------------------------------
opt = Adam(lr = 0.001)

model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
model.add(GlobalMaxPooling1D())
#model.add(Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.3)))
#model.add(Flatten())
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
#--------------------------------------------*Model Train*--------------------------------------------------------------
checkpoint = ModelCheckpoint(os.path.join(best_weights_store_dir, 'LSTM_dist_best_weihts.bin'), monitor="val_loss", save_best_only=True)
rate_shed = LearningRateScheduler(step_decay)
CALLBACKS = [checkpoint]

print("[INFO]: Training....")
#history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xval, yval))

history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=CALLBACKS)

plt.figure()
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")
plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(plot_out_dir, 'print_LSTM_dist_Loss_and_Accuracy.png'))
plt.show()


print("[INFO]: Serializing network...")
model.save(os.path.join(basedir_model_save, 'LSTM_dist_weights.bin'))

