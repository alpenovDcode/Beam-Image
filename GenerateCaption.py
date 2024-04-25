
from PIL import Image
import matplotlib.pyplot as plt

import re

import numpy as np
import matplotlib.pyplot as plt

# Add the following two lines to your code, to have ClearML automatically log your experiment
from clearml import Task


# Create a plot using matplotlib, or you can also use plotly
import os
import pickle
import numpy as np
from keras.src.ops import NotEqual
from keras.src.saving import load_model, custom_object_scope
from nltk.tag import mapping
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

from main import idx_to_word

BASE_DIR = 'C:/Users/Индира/PycharmProjects/ML/datasets'
WORKING_DIR = 'C:/Users/Индира/PycharmProjects/ML/datasets'

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
# создание отображения изображения в подписи
mapping = {}
# обработка строк
for line in tqdm(captions_doc.split('\n')):
    # разделение строки по запятой (,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # удаление расширения из идентификатора изображения
    image_id = image_id.split('.')[0]
    # преобразование списка подписей в строку
    caption = " ".join(caption)
    # создание списка, если это необходимо
    if image_id not in mapping:
        mapping[image_id] = []
    # сохранение подписи
    mapping[image_id].append(caption)

loaded_model = load_model('C:/Users/Индира/PycharmProjects/ML/datasets/model.keras')

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()  # преобразование в нижний регистр
            caption = re.sub(r'[^a-zA-Z ]+', '', caption)  # удаление цифр и специальных символов, кроме пробела
            caption = re.sub(r'\s+', ' ',
                             caption).strip()  # удаление лишних пробелов и пробелов в начале и конце строки
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption

# предварительная обработка текста
clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

print(all_captions[:10]) # пример обработанных подписей

# токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# определение максимальной длины доступной подписи
max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


def predict_caption(model, image, tokenizer, max_length):
    # добавление начального тега для процесса генерации
    in_text = 'startseq'
    # итерация по максимальной длине последовательности
    for i in range(max_length):
        # кодирование входной последовательности
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # дополнение последовательности
        sequence = pad_sequences([sequence], max_length)
        # предсказание следующего слова
        yhat = model.predict([image, sequence], verbose=0)
        # получение индекса с высокой вероятностью
        yhat = np.argmax(yhat)
        # преобразование индекса в слово
        word = idx_to_word(yhat, tokenizer)
        # остановка, если слово не найдено
        if word is None:
            break
        # добавление слова к входу для генерации следующего слова
        in_text += " " + word
        # остановка, если достигнут конечный тег
        if word == 'endseq':
            break

    return in_text

def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(loaded_model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)

print(generate_caption("41999070_838089137e.jpg"))
