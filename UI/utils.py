import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Создает входные файлы для обучения, проверки и тестирования данных.

    :параметр dataset: имя набора данных, одно из "coco", "flickr8k", "flickr30k"
    :параметр karpathy_json_path: путь к файлу Karpathy JSON с разделениями и подписями
    :параметр image_folder: папка с загруженными изображениями
    :параметр captions_per_image: количество подписей для каждого изображения
    ::параметр min_word_freq: слова, встречающиеся реже, чем это пороговое значение, помечаются как <нежелательная> научная информация
    :параметр output_folder: папка для сохранения файлов
    :параметр max_len: не используйте подписи длиннее этой длины
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Считывание путей к изображениям и подписей к каждому изображению
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Создание перечня слов
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Создание base/root имени для всех выходных файлов
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            h.attrs['captions_per_image'] = captions_per_image

            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                assert len(captions) == captions_per_image

                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                for j, c in enumerate(captions):
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Заполняет тензор вложения значениями из равномерного распределения.

    :параметр вложения: тензор вложения
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Создает тензор вложения для указанной карты слов для загрузки в модель.

    :параметр emb_file: файл, содержащий вложения (сохраненный в формате GloVe)
    :параметр word_map: карта слов
    :return: вложения в том же порядке, что и слова в карте слов, размерность вложений
    """

    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Обрезает градиенты, вычисленные при обратном распространении, чтобы избежать увеличения количества градиентов.

    :параметр optimizer: оптимизатор для обрезаемых градиентов
    :параметр grad_clip: значение обрезки
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """Сохраняет контрольную точку модели.

    ::param data_name: базовое имя обработанного набора данных
    :param epoch: номер эпохи
    :param epochs_since_improvement: количество эпох с момента последнего улучшения в оценке BLEU-4
    :param encoder: модель кодера
    :param decoder: модель декодера
    :param encoder_optimizer: оптимизатор для обновления весов кодера при точной настройке
    :param decoder_optimizer: оптимизатор для обновления весовых коэффициентов декодера
    :param bleu4: оценка проверки BLEU-4 для этой эпохи
    :param is_best: является ли эта контрольная точка лучшей на данный момент?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Уменьшает скорость обучения на заданный коэффициент.

    :параметр optimizer: оптимизатор, скорость обучения которого должна быть уменьшена.
    :параметр shrink_factor: коэффициент в интервале (0, 1), на который умножается скорость обучения.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Вычисляет максимальную точность k на основе прогнозируемых и истинных меток.

    :значения параметров: оценки из модели
    :целевые значения параметров: истинные метки
    :параметр k: k с точностью до k
    :возвращает значение с точностью до k
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
