import os
import random
dataset_path = "./"
shuffle_datset = False

images_path = os.path.join(dataset_path, "annotations/train_xmls")

train_path = os.path.join(dataset_path, "annotations/trainval.txt")

images_name = os.listdir(images_path)
if shuffle_datset:
    random.shuffle(images_name)
print("Top of {} number".format(len(images_name)))
train_list = images_name

with open(train_path, "w+") as f:
    for train_name in train_list:
        f.writelines("{}\n".format(train_name[:train_name.find(".xml")]))

