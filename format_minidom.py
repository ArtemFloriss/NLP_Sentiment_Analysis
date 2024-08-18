import xml.dom.minidom
import re

# Путь к вашему XML-файлу
input_file_path = 'book_unlabeled.xml'
output_file_path = 'book_formatted.xml'

# Чтение XML-контента из файла
with open(input_file_path, 'r', encoding='utf-8') as file:
    xml_content = file.read()

# Замена всех символов & на &amp;, чтобы избежать проблем с парсингом
xml_content = xml_content.replace('&', '&amp;')

# Вставка отступов внутри содержимого тегов
def add_indent(content):
    lines = content.splitlines()
    indent_level = 0
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('</'):
            indent_level -= 1
        result.append('    ' * indent_level + stripped)
        if stripped.startswith('<') and not stripped.startswith('</') and not stripped.endswith('/>'):
            indent_level += 1
    return '\n'.join(result)

# Обработка и форматирование XML-контента
formatted_xml = add_indent(xml_content)

# Удаление лишних пустых строк
formatted_xml = re.sub(r'\n\s*\n', '\n', formatted_xml)

# Сохранение отформатированного XML в файл
with open(output_file_path, 'w', encoding='utf-8') as formatted_file:
    formatted_file.write(formatted_xml)

print(f"Форматированный файл сохранен по адресу: {output_file_path}")