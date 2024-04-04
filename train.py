import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Параметры данных
data_folder = './data/Flickr8k/'  # папка с файлами данных, сохраненными скриптом create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # базовое имя, общее для файлов данных

# Параметры модели
emb_dim = 512  # размерность векторов слов
attention_dim = 512  # размерность линейных слоев внимания
decoder_dim = 512  # размерность декодера RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # устройство для модели и тензоров PyTorch
cudnn.benchmark = True  # устанавливается в true, только если размер входных данных модели фиксированный; в противном случае большие вычислительные затраты

# Параметры обучения
start_epoch = 0
epochs = 120  # количество эпох обучения (если ранняя остановка не сработает)
epochs_since_improvement = 0  # отслеживает количество эпох без улучшений в валидационном BLEU
batch_size = 32
workers = 1  # для загрузки данных; в настоящее время работает только с h5py
encoder_lr = 1e-4  # скорость обучения для кодировщика, если производится дообучение
decoder_lr = 4e-4  # скорость обучения для декодера
grad_clip = 5.  # обрезка градиентов по абсолютному значению
alpha_c = 1.  # параметр регуляризации для "двойного стохастического внимания", как в статье
best_bleu4 = 0.  # текущий лучший BLEU-4
print_freq = 100  # частота вывода статистики обучения/валидации каждые __ пакетов
fine_tune_encoder = True  # дообучение кодировщика?
checkpoint = './BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # путь к чекпоинту, None, если нет

def main():
    """
    Обучение и валидация.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Чтение карты слов
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Инициализация / загрузка чекпоинта
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Перемещение на GPU, если доступно
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Функция потерь
    criterion = nn.CrossEntropyLoss().to(device)

    # Специальные загрузчики данных
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Эпохи
    for epoch in range(start_epoch, epochs):

        # Уменьшение скорости обучения, если нет улучшений в течение 8 последовательных эпох, и прекращение обучения после 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # Обучение за одну эпоху
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # Валидация за одну эпоху
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Проверка наличия улучшений
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nКоличество эпох с последнего улучшения: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Сохранение чекпоинта
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Выполняет обучение за одну эпоху.

    :param train_loader: загрузчик данных для обучения
    :param encoder: модель кодировщика
    :param decoder: модель декодера
    :param criterion: слой потерь
    :param encoder_optimizer: оптимизатор для обновления весов кодировщика (если производится дообучение)
    :param decoder_optimizer: оптимизатор для обновления весов декодера
    :param epoch: номер эпохи
    """

    decoder.train()  # режим обучения (используется dropout и batchnorm)
    encoder.train()

    batch_time = AverageMeter()  # время прямого и обратного распространения
    data_time = AverageMeter()  # время загрузки данных
    losses = AverageMeter()  # потери (на слово, декодированное)
    top5accs = AverageMeter()  # точность top-5

    start = time.time()

    # Пакеты
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Перемещение на GPU, если доступно
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Прямое распространение
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Поскольку декодирование начинается с <start>, цели - это все слова после <start> до <end>
        targets = caps_sorted[:, 1:]

        # Удаление временных шагов, которые мы не декодировали, или которые являются паддингами
        # pack_padded_sequence - простой трюк для этого
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Расчет потерь
        loss = criterion(scores, targets)

        # Добавление регуляризации двойного стохастического внимания
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Обратное распространение
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Обрезка градиентов
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Обновление весов
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Отслеживание метрик
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Вывод статуса
        if i % print_freq == 0:
            print('Эпоха: [{0}][{1}/{2}]\t'
                  'Время Пакета {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Время Загрузки Данных {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Потери {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Точность Top-5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

def validate(val_loader, encoder, decoder, criterion):
    """
    Выполняет валидацию за одну эпоху.

    :param val_loader: загрузчик данных для валидации.
    :param encoder: модель кодировщика
    :param decoder: модель декодера
    :param criterion: слой потерь
    :return: BLEU-4 оценка
    """
    decoder.eval()  # режим оценки (без dropout или batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # ссылки (истинные подписи) для расчета оценки BLEU-4
    hypotheses = list()  # гипотезы (предсказания)

    # явное отключение расчета градиента, чтобы избежать ошибки CUDA по памяти
    # решает проблему #57
    with torch.no_grad():
        # Пакеты
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Перемещение на устройство, если доступно
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Прямое распространение
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Поскольку декодирование начинается с <start>, цели - это все слова после <start> до <end>
            targets = caps_sorted[:, 1:]

            # Удаление временных шагов, которые мы не декодировали, или которые являются паддингами
            # pack_padded_sequence - простой трюк для этого
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Расчет потерь
            loss = criterion(scores, targets)

            # Добавление регуляризации двойного стохастического внимания
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Отслеживание метрик
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Валидация: [{0}/{1}]\t'
                      'Время Пакета {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Потери {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Точность Top-5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Сохранение ссылок (истинных подписей) и гипотезы (предсказания) для каждого изображения
            # Если для n изображений у нас есть n гипотез и ссылки a, b, c... для каждого изображения, нам нужно -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # Ссылки
            allcaps = allcaps[sort_ind]  # потому что изображения были отсортированы в декодере
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # удаляем <start> и паддинги
                references.append(img_captions)

            # Гипотезы
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # удаляем паддинги
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Расчет оценок BLEU-4
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * ПОТЕРИ - {loss.avg:.3f}, ТОЧНОСТЬ TOP-5 - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
