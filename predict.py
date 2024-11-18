import os

import torch

from MyLLM.train import seq_length
from model import CharTransformerModel

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функция загрузки модели
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_params = {
        'vocab_size': len(checkpoint['stoi']),
        'embed_size': checkpoint['embed_size'],
        'num_heads': checkpoint['num_heads'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_layers': checkpoint['num_layers'],
        'dropout': checkpoint['dropout']
    }
    model = CharTransformerModel(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['stoi'], checkpoint['itos']

# Загрузка последнего чекпоинта
import glob
checkpoint_list = glob.glob('checkpoints/model_epoch_1.pth')
latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
model, stoi, itos = load_checkpoint(latest_checkpoint)
vocab_size = len(stoi)

# Генерация текста
def generate_text(model, start_text, stoi, itos, max_length=200, temperature=1.0):
    model.eval()
    input_ids = [stoi.get(ch, stoi[' ']) for ch in start_text]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)
    generated_text = start_text

    with torch.no_grad():
        for _ in range(max_length):
            input_seq = input_ids[-seq_length:] if len(input_ids) > seq_length else input_ids
            output = model(input_seq)
            logits = output[-1, 0, :] / temperature
            probs = torch.softmax(logits, dim=0)
            next_char_id = torch.multinomial(probs, num_samples=1).item()
            next_char = itos[next_char_id]
            generated_text += next_char

            # Добавляем следующий символ в последовательность
            input_ids = torch.cat([input_ids, torch.tensor([[next_char_id]], device=device)], dim=0)

    return generated_text

# Пример использования
if __name__ == '__main__':
    start_text = input("Введите начальный текст: ")
    generated_text = generate_text(model, start_text, stoi, itos, max_length=500, temperature=0.8)
    print("Сгенерированный текст:")
    print(generated_text)