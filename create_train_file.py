import re
import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from lxml import etree
#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

########################################################################################################################
def clean_xml_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Удаляем некорректные символы
    content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', '', content)

    return content


def process_file(input_file_path, is_positive):
    content = clean_xml_content(input_file_path)
    try:
        root = etree.fromstring(content)
    except etree.XMLSyntaxError as e:
        print(f"Ошибка парсинга XML в файле {input_file_path}: {e}")
        return []

    reviews = []
    for review in root.findall('review'):
        try:
            helpful = review.find('helpful').text.strip() if review.find('helpful') is not None else None
            rating = review.find('rating').text.strip() if review.find('rating') is not None else None
            review_text = review.find('review_text').text.strip() if review.find('review_text') is not None else None

            if helpful and rating and review_text:
                new_review = etree.Element('review')
                is_positive_elem = etree.SubElement(new_review, 'is_positive')
                is_positive_elem.text = '1' if is_positive else '0'

                for tag, text in [('helpful', helpful), ('rating', rating), ('review_text', review_text)]:
                    new_elem = etree.SubElement(new_review, tag)
                    new_elem.text = text

                reviews.append(new_review)
        except Exception as e:
            print(f"Ошибка обработки отзыва: {e}")

    return reviews
def create_emotion_file(output_file_path, positive_file, negative_file):
    positive_reviews = process_file(positive_file, is_positive=True)
    negative_reviews = process_file(negative_file, is_positive=False)

    # Объединяем и перемешиваем отзывы
    all_reviews = positive_reviews + negative_reviews
    random.shuffle(all_reviews)

    annotation = etree.Element('annotation')
    for review in all_reviews:
        annotation.append(review)
    # Применяем форматирование с отступами
    xml_str = etree.tostring(annotation, pretty_print=True, encoding='utf-8', xml_declaration=True)

    with open(output_file_path, 'wb') as f:
        f.write(xml_str)

    print(f"Файл '{output_file_path}' успешно создан.")


# Путь к корневой папке
root_dir = ROOT_DIR

# Пути к файлам
positive_file = os.path.join(root_dir, 'positive_format_all.xml')
negative_file = os.path.join(root_dir, 'negative_format_all.xml')
output_file = os.path.join(root_dir, 'emotion_format_all.xml')

# Создаем объединенный файл
create_emotion_file(output_file, positive_file, negative_file)
