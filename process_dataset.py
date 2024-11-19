import os

import pandas as pd
import tqdm

def combine_c3_to_text(csv_filepath, output_filepath):
    """
    Читает CSV файл, извлекает значения из столбца 'C3',
    объединяет их в один текстовый файл, разделяя записи '. '.
    Отображает прогрессбар.

    Args:
        csv_filepath: Путь к CSV файлу.
        output_filepath: Путь к выходному текстовому файлу.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Ошибка: Файл {csv_filepath} не найден.")
        return
    except pd.errors.ParserError:
        print(f"Ошибка: Не удалось прочитать CSV файл {csv_filepath}. Проверьте формат файла.")
        return

    c3_values = df['text'].astype(str).tolist()

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for i in tqdm.tqdm(range(len(c3_values)), desc="Запись в файл"):  # Прогрессбар
            outfile.write(c3_values[i])
            if i < len(c3_values) - 1:  # Добавляем разделитель, кроме последнего элемента
                outfile.write('. ')

    print(f"Тексты из столбца C3 успешно объединены в файл: {output_filepath}")



# Пример использования:
##csv_file = 'dataset/lenta-ru-news.csv'
##output_file = 'combined_c3.txt'
##combine_c3_to_text(csv_file, output_file)




def split_text_file(input_filepath, output_filepath, desired_size):
    """
    Копирует часть текстового файла, чтобы он соответствовал нужному размеру.

    Args:
        input_filepath: Путь к исходному файлу.
        output_filepath: Путь для сохранения нового файла.
        desired_size: Желаемый размер выходного файла в байтах.
    """

    # Проверяем, существует ли исходный файл
    if not os.path.exists(input_filepath):
        print(f"Ошибка: Исходный файл не найден: {input_filepath}")
        return

    # Получаем размер исходного файла
    original_size = os.path.getsize(input_filepath)

    # Проверяем, достаточен ли размер исходного файла
    if original_size < desired_size:
        print(f"Ошибка: Исходный файл меньше желаемого размера ({original_size} < {desired_size}).")
        return

    # Открываем исходный файл в режиме чтения в двоичном формате
    with open(input_filepath, 'rb') as infile:
        # Создаем новый файл для записи
        with open(output_filepath, 'wb') as outfile:
            # Копируем данные порциями, используя tqdm для отображения прогресса
            bytes_copied = 0
            with tqdm.tqdm(total=desired_size, unit='B', unit_scale=True, desc="Копирование файла") as pbar:
                while bytes_copied < desired_size:
                    # Определяем размер порции для чтения
                    chunk_size = min(1024 * 1024, desired_size - bytes_copied)  # Читаем максимум 1 МБ за раз

                    # Читаем порцию данных из исходного файла
                    chunk = infile.read(chunk_size)

                    # Записываем порцию данных в новый файл
                    outfile.write(chunk)

                    # Обновляем счетчик скопированных байтов и прогресс-бар
                    bytes_copied += len(chunk)
                    pbar.update(len(chunk))

    print(f"Файл успешно скопирован в {output_filepath} с размером {desired_size} байт.")

# Пример использования:
input_file = "combined_c3.txt"  # Замените на путь к вашему файлу
output_file = "dataset/combined_c3_part.txt"   # Замените на желаемый путь сохранения
desired_size = 50 * 1024 * 1024  # 10 МБ (замените на нужный размер)

split_text_file(input_file, output_file, desired_size)