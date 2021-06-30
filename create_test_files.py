import shutil, random, os, math
dirpath = 'E:\\ProjekatORI\\Project\\Traffic_Sign_Classification\\data'
destDirectory = 'E:\\ProjekatORI\\Project\\Traffic_Sign_Classification\\test_data'

classes = []

for classname in os.listdir(dirpath):
    classes.append(classname)

for classname in classes:
    origin_folder = dirpath + "\\" + classname
    dest_folder = destDirectory + "\\" + classname

    number_of_imgs = len(os.listdir(origin_folder))
    number_to_pick = math.floor(number_of_imgs * 0.1)  # 10% testnih

    filenames = random.sample(os.listdir(origin_folder), number_to_pick)

    for filename in filenames:
        path_from = origin_folder + "\\" + filename
        path_to_move = dest_folder + "\\" + filename
        shutil.move(path_from, path_to_move)
