import os
import re
from tqdm import tqdm


def analyze_file(filepath):
    """
    Анализирует текстовый файл и возвращает количество слов и предложений.

    Args:
        filepath: Путь к текстовому файлу.

    Returns:
        Кортеж (word_count, sentence_count) или None, если файл не существует или не является текстовым.
    """

    if not os.path.exists(filepath) or not filepath.lower().endswith(".txt"):
        print(f"Ошибка: Файл '{filepath}' не найден или не является текстовым.")
        return None

    word_count = 0
    sentence_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as file:  # Добавлена обработка кодировки
            # Определяем размер файла для tqdm
            file_size = os.path.getsize(filepath)
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Обработка файла") as pbar:
                for line in file:
                    pbar.update(len(line.encode('utf-8')))  # Обновляем tqdm с учетом кодировки

                    # Подсчет слов (более надежное регулярное выражение)
                    words = re.findall(r'\b\S+\b', line)  # Исключает знаки препинания из слов
                    word_count += len(words)

                    # Подсчет предложений (более надежное регулярное выражение)
                    sentences = re.split(r'[.!?;]+', line)
                    sentences = [s for s in sentences if s.strip()]  # Исключаем пустые предложения после split
                    sentence_count += len(sentences)


    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

    return word_count, sentence_count


# Пример использования
filepath = "dataset/combined_c3_part.txt"  # Замените на путь к вашему файлу
result = analyze_file(filepath)

if result:
    word_count, sentence_count = result
    print(f"Количество слов: {word_count}")
    print(f"Количество предложений: {sentence_count}")