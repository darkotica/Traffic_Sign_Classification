import pandas as pd
import os
import math
from PIL import Image
from pathlib import Path


path_03 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00003\\'
path_07 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00007\\'
path_09 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00009\\'
path_11 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00011\\'
path_12 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00012\\'
path_13 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00013\\'
path_14 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00014\\'
path_17 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00017\\'
path_18 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00018\\'
path_25 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00025\\'
path_31 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00031\\'
path_35 = 'E:\\ProjekatORI\\Traffic sign classification - dataset\\Sign classes\\00035\\'

paths_folders = [path_03, path_07, path_09,
                 path_11, path_12, path_13,
                 path_14, path_17, path_18,
                 path_25, path_31, path_35]

paths_csv = [path_03 + 'GT-00003.csv', path_07 + 'GT-00007.csv', path_09 + 'GT-00009.csv',
             path_11 + 'GT-00011.csv', path_12 + 'GT-00012.csv', path_13 + 'GT-00013.csv',
             path_14 + 'GT-00014.csv', path_17 + 'GT-00017.csv', path_18 + 'GT-00018.csv',
             path_25 + 'GT-00025.csv', path_31 + 'GT-00031.csv', path_35 + 'GT-00035.csv']


if __name__ == '__main__':
    num_of_pics = 0
    total_w = 0
    total_h = 0

    content_list = []
    for path in paths_csv:
        content = pd.read_csv(path, header=0, sep=';')
        content_list.append(content)
        total_content = len(content.index)
        training = math.floor(0.7*total_content)
        validation = math.floor(0.2*total_content)
        test = total_content - training - validation
        print(path + "\n\ttotal: {0} | training: {1} | validation: {2} | test: {3}\n\n".format(total_content,
              training, validation, test))
        total_w += content['Width'].sum()
        total_h += content['Height'].sum()
        num_of_pics += total_content

    avg_w = total_w / num_of_pics
    avg_h = total_h / num_of_pics
    print(avg_w)  # 53.96114146933819
    print(avg_h)  # 52.85173041894353
    # mozemo uzeti recimo 55x55 da bude ulaz

    # print("\n###starting with resizing###\n")
    # content_count = 0
    # for path_folder in paths_folders:
    #     resized_folder = path_folder + "resized_images"
    #     Path(resized_folder).mkdir(parents=True, exist_ok=True)
    #     row_count = 0
    #     for file in os.listdir(path_folder):
    #         filename = os.fsdecode(file)
    #         if not filename.endswith(".ppm"):
    #             continue
    #         image = Image.open(path_folder+filename)
    #         x1 = content_list[content_count].loc[row_count, 'Roi.X1']
    #         y1 = content_list[content_count].loc[row_count, 'Roi.Y1']
    #         x2 = content_list[content_count].loc[row_count, 'Roi.X2']
    #         y2 = content_list[content_count].loc[row_count, 'Roi.Y2']
    #
    #         cropped_img = image.crop((x1, y1, x2, y2))
    #         new_image = cropped_img.resize((55, 55))
    #         new_image.save(resized_folder + "\\" + filename)
    #         print("Finished with " + filename + " in folder " + path_folder)
    #         row_count += 1
    #
    #     content_count += 1




