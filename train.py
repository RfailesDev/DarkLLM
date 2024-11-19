# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharTransformerModel
import numpy as np
from torch import amp
import random


# Класс Dataset, который читает данные по мере необходимости
class TextDataset(Dataset):
    def __init__(self, dataset_path, stoi, seq_length):
        self.stoi = stoi
        self.seq_length = seq_length
        self.data = []

        # Предварительная загрузка всех данных в память
        for filename in os.listdir(dataset_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dataset_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        self.data.extend([self.stoi.get(ch, self.stoi[' ']) for ch in text])
                except UnicodeDecodeError:
                    print(f"Файл {file_path} содержит некорректные символы и будет пропущен.")

        self.data = torch.tensor(self.data, dtype=torch.long)
        self.total_sequences = len(self.data) - self.seq_length

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return input_seq, target_seq


def build_vocab(dataset_path):
    chars = set()
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(dataset_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(1024 * 1024)  # Чтение чанками по 1МБ
                    if not chunk:
                        break
                    chars.update(chunk)
    chars = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, chars


def train(model, dataloader, criterion, optimizer, num_epochs, device, stoi, itos, embed_size, num_heads, hidden_dim, num_layers, dropout):
    model.train()
    scaler = amp.GradScaler()  # Исправлено согласно FutureWarning

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            # Перестановка размерностей: (batch_size, seq_length) уже соответствует batch_first=True
            input_seq = input_seq.to(device, non_blocking=True)
            target_seq = target_seq.to(device, non_blocking=True).long()  # Убедимся, что тип Long

            optimizer.zero_grad()
            str_device = torch.device("cuda" if torch.cuda.is_available() else "cpu").__str__()
            with amp.autocast(device_type=str_device):
                output = model(input_seq)  # output: (batch_size, seq_length, vocab_size)
                # Изменение формы для CrossEntropyLoss: (batch_size * seq_length, vocab_size)
                output = output.view(-1, output.size(-1))
                # Цели: (batch_size * seq_length)
                target = target_seq.view(-1)

                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Для градиентного клиппинга
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Градиентный клиппинг
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Эпоха [{epoch + 1}/{num_epochs}], Батч [{batch_idx}/{len(dataloader)}], Потеря: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Эпоха [{epoch + 1}/{num_epochs}], Средняя потеря: {avg_loss:.4f}")

        # Сохранение чекпоинта после каждой эпохи
        checkpoint_path = f'checkpoints/model_epoch_{epoch + 1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'stoi': stoi,
            'itos': itos,
            'embed_size': embed_size,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }, checkpoint_path)
        print(f"Чекпоинт сохранен: {checkpoint_path}")

    # Сохранение финальной модели
    torch.save(model.state_dict(), 'model_final.pth')
    print("Финальная модель сохранена: model_final.pth")


def main():
    # Фиксация случайных зерен для воспроизводимости
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Параметры модели и обучения
    embed_size = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 4
    dropout = 0.1
    batch_size = 512 #512
    seq_length = 128
    num_epochs = 50
    learning_rate = 1e-4

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Построение словаря символов без загрузки всего датасета в оперативную память
    stoi, itos, chars = build_vocab('dataset')
    vocab_size = len(chars)
    print(f"Размер словаря: {vocab_size}")

    if vocab_size == 0:
        raise ValueError("Словарь пуст. Проверьте содержимое директории 'dataset'.")

    # Создание DataLoader
    dataset = TextDataset('dataset', stoi, seq_length)
    if len(dataset) == 0:
        raise ValueError("Датасет не содержит последовательностей для обучения.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=4)

    # Инициализация модели
    model = CharTransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Запуск обучения
    train(model, dataloader, criterion, optimizer, num_epochs, device,
          stoi, itos, embed_size, num_heads, hidden_dim, num_layers, dropout)


if __name__ == '__main__':
    main()