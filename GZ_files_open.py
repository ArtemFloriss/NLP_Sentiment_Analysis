import gzip

# Путь к файлу
file_path = 'book.unlabeled.gz'

# Открытие и чтение содержимого файла
with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    content = file.read()

# Вывод первых 1000 символов содержимого
print(content[:1000])