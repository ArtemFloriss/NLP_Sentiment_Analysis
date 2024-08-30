import os

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
from tensorflow.keras import models

import json
#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

MAX_FEATURES = 10000
MAX_SENTENCE_LENGTH = 200

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 12

VOCAB_SIZE = MAX_FEATURES
########################################################################################################################
##################################################*PATHS*###############################################################
BASE_DICTS = 'Convert_dicts'
basedir_model_save = 'LSTM_bin'
best_weights_store_dir = 'Best_weights'
TF_SERVING_MODEL_DIR = 'tf_serving_model'
################################################*FUNCTIONS*#############################################################
# Функция для загрузки словаря из JSON
def load_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
########################################################################################################################
# Загрузка словаря
word2index = load_dict_from_json(os.path.join(BASE_DICTS,'word2index.json'))

VOCAB_SIZE = len(word2index)

# Воссоздаем архитектуру модели
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(Bidirectional(LSTM(int(HIDDEN_LAYER_SIZE), return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# Загрузка весов из bin-файла
model_path = os.path.join(best_weights_store_dir, 'LSTM_dist_best_weihts.bin')
model.load_weights(model_path)
#model = models.load_model(model_path)

# Преобразование модели в TensorFlow Serving модель
@tf.function(input_signature=[tf.TensorSpec(shape=[None, MAX_SENTENCE_LENGTH], dtype=tf.int32)])
def serving_fn(input_data):
    return {'output': model(input_data)}

# Сохранение модели в формате SavedModel для TensorFlow Serving
tf.saved_model.save(model, os.path.join(TF_SERVING_MODEL_DIR,'sentiment_model'), signatures={'serving_default': serving_fn})

print("TF Serving model creation is OK")