import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    Класс Dataset PyTorch, который используется в DataLoader PyTorch для создания пакетов.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: папка, где хранятся файлы данных
        :param data_name: базовое имя обработанных наборов данных
        :param split: раздел, один из 'TRAIN', 'VAL' или 'TEST'
        :param transform: пайплайн трансформации изображений
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Открытие hdf5 файла, где хранятся изображения
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Подписи на изображение
        self.cpi = self.h.attrs['captions_per_image']

        # Загрузка закодированных подписей (полностью в память)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Загрузка длин подписей (полностью в память)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Пайплайн трансформации PyTorch для изображения (нормализация и т.д.)
        self.transform = transform

        # Общее количество данных
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Напоминаем, N-ая подпись соответствует (N // captions_per_image)-ому изображению
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # Для валидации или тестирования, также возвращаем все подписи 'captions_per_image', чтобы найти оценку BLEU-4
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
