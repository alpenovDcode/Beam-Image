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

BASE_DIR = 'C:/Users/Индира/PycharmProjects/ML/datasets'
WORKING_DIR = 'C:/Users/Индира/PycharmProjects/ML/datasets'



model = VGG16()

model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

print(model.summary())



features = {}
directory = os.path.join(BASE_DIR, 'Images')
'''
for img_name in tqdm(os.listdir(directory)):

    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))

    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature

# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
'''
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)



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

print(len(mapping) ) # Вывод длины словаря, что показывает количество уникальных изображений


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


#        X                   y
# startseq                   девочка
# startseq девочка           входит
# startseq девочка входит    в
# ...........
# startseq девочка входит в деревянное здание      endseq
# создание генератора данных для получения данных
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # цикл по изображениям
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # обработка каждой подписи
            for caption in captions:
                # кодирование последовательности
                seq = tokenizer.texts_to_sequences([caption])[0]
                # разделение последовательности на пары X, y
                for i in range(1, len(seq)):
                    # разделение на входные и выходные пары
                    in_seq, out_seq = seq[:i], seq[i]
                    # дополнение входной последовательности
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # кодирование выходной последовательности
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    # сохранение последовательностей
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

# Модель кодировщика
# Слои признаков изображения
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# Слои признаков последовательности
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Модель декодировщика
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')



def make_dataset(data_keys, batch_size):
    def generator():
        # Повторяем логику вашего генератора data_generator
        for key in data_keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield (features[key][0], in_seq), out_seq

    output_signature = (
        (
            tf.TensorSpec(shape=(4096,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(batch_size)
'''
# обучение модели
epochs =40
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # Создаём Dataset для обучающего набора
    train_dataset = make_dataset(train, batch_size)
    # Обучаем модель
    model.fit(train_dataset, epochs=1, steps_per_epoch=steps, verbose=1)
# сохранение модели
model.save(WORKING_DIR + '/model1.keras')
'''
from tqdm import tqdm

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# генерация подписи к изображению
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


from nltk.translate.bleu_score import corpus_bleu

# validate with test data
actual, predicted = list(), list()

loaded_model = load_model('C:/Users/Индира/PycharmProjects/ML/datasets/model1.keras')
'''
for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(loaded_model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)

# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
'''
'''


from PIL import Image
import matplotlib.pyplot as plt
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

print(generate_caption("667626_18933d713e.jpg"))
'''
