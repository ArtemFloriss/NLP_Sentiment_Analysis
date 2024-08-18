import re
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

# Путь к вашему XML-файлу
input_file_path = 'book_unlabeled.xml'
output_file_path = 'book_formatted.xml'

# Загрузка содержимого файла
with open(input_file_path, 'r', encoding='utf-8') as file:
    xml_content = file.read()

# Замена всех символов & на &amp;, кроме случаев, когда & уже является частью сущности (&amp;, &lt; и т.д.)
xml_content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', xml_content)

# Удаление невалидных символов (например, управляющие символы)
# Этот код оставляет только допустимые символы Unicode, которые могут быть частью XML
xml_content = ''.join(c for c in xml_content if c.isprintable() or c in '\t\n\r')

# Парсинг XML после замены
try:
    root = ET.fromstring(xml_content)

    # Функция для добавления отступов
    def pretty_print(element, level=0):
        indent = '    ' * level
        if len(element):
            if not element.text or not element.text.strip():
                element.text = '\n' + indent + '    '
            if not element.tail or not element.tail.strip():
                element.tail = '\n' + indent
            for elem in element:
                pretty_print(elem, level+1)
            if not element.tail or not element.tail.strip():
                element.tail = '\n' + indent
        else:
            if level and (not element.tail or not element.tail.strip()):
                element.tail = '\n' + indent

    pretty_print(root)

    # Преобразование дерева в строку
    formatted_xml = ET.tostring(root, encoding='unicode')

    # Удаление лишних пустых строк
    formatted_xml = re.sub(r'\n\s*\n', '\n', formatted_xml)

    # Сохранение форматированного XML в файл
    with open(output_file_path, 'w', encoding='utf-8') as formatted_file:
        formatted_file.write(formatted_xml)

    print(f"Форматированный файл сохранен по адресу: {output_file_path}")



except ET.ParseError as e:
    print(f"Ошибка парсинга XML: {e}")