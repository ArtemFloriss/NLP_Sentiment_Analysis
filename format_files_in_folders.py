import os
import re

#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

########################################################################################################################
def ensure_annotation_tags(content):
    # Проверяем наличие открывающего тега <annotation> в начале файла
    if not content.strip().startswith('<annotation>'):
        content = '<annotation>\n' + content

    # Проверяем наличие закрывающего тега </annotation> в конце файла
    if not content.strip().endswith('</annotation>'):
        content = content + '\n</annotation>'

    return content
#-----------------------------------------------------------------------------------------------------------------------
def format_xml_file(input_file_path, output_file_path):
    # Чтение XML-контента из файла
    with open(input_file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Обеспечиваем наличие тегов <annotation> и </annotation>
    xml_content = ensure_annotation_tags(xml_content)

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


# Корневая папка, в которой лежат другие папки с XML-файлами
root_dir = ROOT_DIR

# Перебираем все папки и файлы в корневой директории
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.review.xml'):
            input_file_path = os.path.join(foldername, filename)
            output_file_name = filename.replace('.review', '_formatted')
            output_file_path = os.path.join(foldername, output_file_name)

            # Форматируем XML файл
            format_xml_file(input_file_path, output_file_path)