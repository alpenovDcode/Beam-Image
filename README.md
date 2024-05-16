# Проект Image Captions

## Обзор

Этот проект генерирует описательные подписи к изображениям, используя комбинацию предобученной модели ResNet50 для извлечения признаков и нейронной сети на основе LSTM для генерации текста. Используемый набор данных - Flickr8k, который содержит 8000 изображений и соответствующие им подписи.

## Содержание

1. [Установка](#установка)
2. [Подготовка данных](#подготовка-данных)
3. [Извлечение признаков](#извлечение-признаков)
4. [Обработка текста](#обработка-текста)
5. [Обучение модели](#обучение-модели)
6. [Использование](#использование)
7. [Визуализация](#визуализация)
8. [Проверка и оценка модели](#проверка-и-оценка-модели)

## Установка

Для начала необходимо установить необходимые библиотеки Python. Вы можете сделать это с помощью pip:

```bash
pip install -r requirements.txt
```

## Подготовка данных

Набор данных Flickr8k загружается и распаковывается с помощью библиотеки `opendatasets`.

```python
import opendatasets as od

od.download("https://www.kaggle.com/datasets/kunalgupta2616/flickr-8k-images-with-captions")
```

## Извлечение признаков

Проект использует предобученную модель ResNet50 для извлечения признаков из изображений. Модель модифицируется для использования предпоследнего слоя.

```python
from keras.applications import ResNet50
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import pickle
from tqdm import tqdm

# Загрузка предобученной модели ResNet50
model = ResNet50(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Директория с изображениями
directory = '/content/flickr-8k-images-with-captions/Images'

# Инициализация словаря для хранения признаков
features = {}

# Извлечение признаков для каждого изображения
for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Сохранение признаков в файл
with open('/content/Features.pkl', 'wb') as f:
    pickle.dump(features, f)
```

## Обработка текста

Подписи загружаются, очищаются и токенизируются для подготовки к обучению.

```python
# Загрузка подписей
with open('/content/flickr-8k-images-with-captions/captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

# Создание отображения идентификатора изображения к подписям
mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Очистка подписей
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

clean(mapping)

# Токенизация подписей
from keras.preprocessing.text import Tokenizer

all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)
```

## Обучение модели

Модель строится с использованием слоя встраивания (embedding), LSTM и плотных слоев для генерации подписей.

```python
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from keras.models import Model

# Разделение данных на обучающую и тестовую выборки
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# Генератор данных
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0

# Построение модели
inputs1 = Input(shape=(2048,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Обучение модели
epochs = 10
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Сохранение модели
model.save('/content/best_model.h5')
```

## Использование

Для генерации подписей для нового изображения, загрузите обученную модель и используйте её для предсказания подписи.

```python
# Функция для преобразования индекса в слово
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Генерация подписи для изображения
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text
```

## Визуализация

Вы можете визуализировать архитектуру модели с помощью `plot_model`.

```python
from keras.utils import plot_model

plot_model(model, show_shapes=True)
```

## Проверка и оценка модели

Модель проверяется на тестовых данных, и оценивается с использованием BLEU-оценок.

```python
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# Проверка модели на тестовых данных
actual, predicted = list(), list()

for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    actual.append(actual_captions)
    predicted.append(y_pred)
 
# Расчет BLEU-оценок
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0,

 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# Генерация и визуализация подписи для конкретного изображения
from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join('/content/flickr-8k-images-with-captions/Images', image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()

generate_caption("101669240_b2d3e7f17b.jpg")
```

## Заключение

Этот проект демонстрирует, как создать систему генерации подписей к изображениям с использованием методов глубокого обучения. Предобученная модель ResNet50 извлекает признаки из изображений, а нейронная сеть на основе LSTM генерирует подписи на основе этих признаков. Полный процесс включает подготовку данных, извлечение признаков, обработку текста, обучение модели и оценку.