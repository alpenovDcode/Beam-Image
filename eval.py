import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# выполняется несколько шагов, необходимых для оценки производительности модели генерации
# подписей к изображениям с использованием алгоритма Beam Search

# Parameters

# В этой папке находятся данные для генерации описаний изображений
data_folder = '/media/ssd/caption data'  # Путь к папке, где сохранены файлы, созданные с помощью create_input_files.py

data_name = 'coco_5_cap_per_img_5_min_word_freq'  # базовое имя обработанных наборов данных

# Путь к файлу, содержащему модель и сохраненные параметры после обучения.
checkpoint = '../BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

# Путь к файлу сопоставления слов с индексами (word map), который был использован при обучении модели.
word_map_file = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

# Определяет устройство, на котором будет выполняться модель: GPU (если доступен) или CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Устанавливает значение True, если размер входных данных в модель фиксирован
cudnn.benchmark = True

# Загрузка модели

# Загружается сохраненный чекпоинт модели с диска
checkpoint = torch.load(checkpoint)

# Из чекпоинта извлекаются декодер и энкодер модели.
decoder = checkpoint['decoder']
# Оба модуля перемещаются на указанное устройство (GPU или CPU) с помощью .to(device).
decoder = decoder.to(device)

# Оба модуля переводятся в режим оценки с помощью .eval()
# говорим нашей модели: "теперь ты должна предсказывать, а не учиться"
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Загрузка слова в индекс (word map):
# Открывается файл word_map_file, содержащий соответствие между словами и их индексами.
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Создается обратное соответствие, где индексы становятся ключами, а слова - значениями
rev_word_map = {v: k for k, v in word_map.items()}

# Вычисляется размер словаря
vocab_size = len(word_map)

# Нормализация изображений:
# нормализация средних и стандартных отклонений,
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Оценка производительности модели для генерации подписей к изображениям с использованием алгоритма Beam Search

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    # CaptionDataset для загрузки тестового набора данных изображений и соответствующих подписей
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    # loader - объяект для доступа к данным

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list() # список списков эталонных описаний
    hypotheses = list() # список сгенерированных описаний

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        # количество альтернативных предсказаний, которые будут рассмотрены на каждом шаге декодирования
        k = beam_size

        # Move to GPU device, if available
        # Перемещает изображение на устройство GPU
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # Исходное изображение передается через кодер для извлечения его важных признаков.
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

        # размерность пространства признаков, в котором закодировано изображение.
        enc_image_size = encoder_out.size(1)

        # размерность признакового пространства, которое выдает кодер.
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        # Переформатирование
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        # Вычисление числа пикселей
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        # размерность признакового пространства, полученного от кодера
        # В результате получается тензор формы (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        # Создается тензор для хранения предыдущих топ-к слов на каждом шаге декодирования
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        # тензор, который будет использоваться для хранения последовательностей слов на каждом шаге декодирования
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        # тензор, который будет использоваться для хранения оценок вероятности для каждой из k последовательностей на текущем шаге декодирования
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        # список для хранения завершенных последовательностей
        complete_seqs = list()
        # список для хранения оценок
        complete_seqs_scores = list()

        # Начало декодирования последовательностей

        # текущий шаг декодирования
        step = 1

        # скрытое состояние (h) и ячейка памяти (c) декодера
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            # генерация векторных представлений для предыдущих слов в последовательности (векторные представления слов)
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            # применение механизма внимания
            # Для каждого слова в текущей последовательности h и для каждой области изображения encoder_out,
            # вычисляется "внимательность" (attention), которая показывает,
            # насколько каждая область изображения важна для генерации следующего слова.
            # внимательность для каждой области изображения размеров (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            # вычисляется весовой коэффициент для каждой области изображения
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            # учесть важность каждой области изображения
            awe = gate * awe

            # векторы эмбеддингов слов и внимания объединяются горизонтально. Получаем ветор признаков
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            # вычисляются оценки для каждого слова в словаре на основе текущего скрытого состояния h.
            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            # добавляются оценки (scores) текущего шага к предыдущим оценкам
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            #  выбираются топ-K оценок и соответствующие слова для продолжения предложения на следующем шаге.
            if step == 1:
                # все K точек имеют одинаковые оценки
                # выбираем самые высокие оценки
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # выбираются топ-K оценок из объединенного списка оценок всех предложений в текущем шаге.
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            # Здесь происходит преобразование развернутых индексов (top_k_words)
            # в реальные индексы в матрице оценок (scores). (двумерн предст)
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            # добавляются новые слова в последовательности. Размерность seqs увеличивается до (s, step+1)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            # Здесь определяются индексы последовательностей, которые до сих пор не достигли символа '<end>'
            # Если след слово в посл-ти != '<end>' -> последовательность остается неполной
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            # проверяется, есть ли завершенные последовательности (те, которые достигли символа '<end>').
            # (k -= len(complete_inds)). Кол-о кандидатов уменьш на кол-о заверш посл-й
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            # если кандидатов не осталось
            if k == 0:
                break

            # оставляются только незавершенные последовательности
            seqs = seqs[incomplete_inds]

            # ставляем только незавершенные последовательности
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            # Предотвратить случай бесконечного выполнения
            if step > 50:
                break
            step += 1

        # Выбираем индекс, который имеет наивысший балл (score)
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        # список со списком слов для каждого изображения
        img_caps = allcaps[0].tolist()

        # удаляем <start>, <end>, <pad>
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads

        # добавляем его в references
        references.append(img_captions)

        # Hypotheses
        # список гипотез
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        # проверка, что кол-о эл-в в 2-х списках одинакого
        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # вычисляет метрику BLEU-4 для сгенерированных описаний изображений.
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))