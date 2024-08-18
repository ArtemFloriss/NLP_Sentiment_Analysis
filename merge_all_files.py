import os

#############################################*PARAMETERS*###############################################################
ROOT_DIR = 'sorted_data_acl'

########################################################################################################################
def merge_files(output_file_path, input_files):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write('<annotation>\n')

        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as infile:
                # Чтение содержимого файла, кроме первой и последней строки (<annotation> и </annotation>)
                lines = infile.readlines()[1:-1]
                outfile.writelines(lines)
                outfile.write('\n')

        outfile.write('</annotation>\n')
#-----------------------------------------------------------------------------------------------------------------------

# Корневая папка, в которой лежат все другие папки с XML-файлами
root_dir = ROOT_DIR

# Списки для хранения путей к файлам для объединения
positive_files = []
negative_files = []
unlabeled_files = []

# Перебираем все папки и файлы в корневой директории
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'positive_formatted.xml':
            positive_files.append(os.path.join(foldername, filename))
        elif filename == 'negative_formatted.xml':
            negative_files.append(os.path.join(foldername, filename))
        elif filename == 'unlabeled_formatted.xml':
            unlabeled_files.append(os.path.join(foldername, filename))

# Объединяем файлы
merge_files(os.path.join(root_dir, 'positive_format_all.xml'), positive_files)
merge_files(os.path.join(root_dir, 'negative_format_all.xml'), negative_files)
merge_files(os.path.join(root_dir, 'unlabeled_format_all.xml'), unlabeled_files)

print("Файлы успешно объединены.")