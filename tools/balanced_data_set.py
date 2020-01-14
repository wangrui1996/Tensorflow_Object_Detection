import os
import cv2
import shutil
from libs.data_augmentation import *


def copy_images_and_xmls(src_images_path, src_xmls_path, dst_images_path, dst_xmls_path):
    src_images_name = os.listdir(src_images_path)
    top_num = len(src_images_name)
    print("Top number {} will be copy".format(top_num))
    index = 1
    for src_image_name in src_images_name:
        file_name = src_image_name.split(".")[0]
        shutil.copy(os.path.join(src_xmls_path, "{}.xml".format(file_name)), os.path.join(dst_xmls_path, "{}.xml".format(file_name)))
        if src_image_name.split(".")[-1] == "jpg":
            shutil.copy(os.path.join(src_images_path, "{}.jpg".format(file_name)), os.path.join(dst_images_path, "{}.jpg".format(file_name)))
        else:
            image = cv2.imread(os.path.join(src_images_path, src_image_name))
            cv2.imwrite(os.path.join(dst_images_path, "{}.jpg".format(file_name)), image)
        if index % 1000 == 0:
            print("{} finished of Top {}".format(index, top_num))
        index = index + 1

def find_sample_file(xmls_path):
    xmls_name = os.listdir(xmls_path)
    include_sample_files_name = []
    for xml_name in xmls_name:
        if is_have_object_in_xml(os.path.join(xmls_path, xml_name)):
            include_sample_files_name.append(xml_name.split(".")[0])
    return include_sample_files_name


def create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

class DataSet:
    def __init__(self, sample_files_name):
        self.sample_files_name = sample_files_name
        self.index = 0
        self.num_samples = len(sample_files_name)

    def get_sample_file_name(self):
        if self.index < self.num_samples:
            file_name = self.sample_files_name[self.index]
            self.index = self.index + 1
        else:
            random.shuffle(self.sample_files_name)
            file_name = self.sample_files_name[0]
            self.index = 1
        return file_name

    def get_image_xml_path(self):
        file_name = self.get_sample_file_name()
        return file_name

if __name__ == '__main__':

    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    thread_num = 50
    lock = multiprocessing.Lock()
    executor = ThreadPoolExecutor(thread_num)

    src_images_path = "./images"
    src_xmls_path = "./annotations/xmls"
    dst_images_path = "./train_images"
    dst_xmls_path = "./annotations/train_xmls"
    create_folder(dst_images_path)
    create_folder(dst_xmls_path)
    copy_images_and_xmls(src_images_path, src_xmls_path, dst_images_path, dst_xmls_path)
    images_path = dst_images_path
    xmls_path = dst_xmls_path
    dataset_files_name = [src_image_name.split(".")[0] for src_image_name in os.listdir(images_path)]
    include_sample_files_name = find_sample_file(xmls_path)
    dataset = DataSet(include_sample_files_name)
    print("Top of {} is have object".format(len(include_sample_files_name)))
    augmentation_numbers = 30000
    name_sub = "add"
    file_index = 1
    progress_index = 0
    def progress(src_image_path, src_xml_path, dst_image_path, dst_xml_path):
        augmentation_one_image(src_image_path, src_xml_path, dst_image_path, dst_xml_path)
#        print(src_image_path, dst_image_path)
        with lock:
            global progress_index
#            print(progress_index)
            progress_index = progress_index + 1
            print('\r', "Pregress: {:>10d}".format(progress_index), end='')

    for _ in range(augmentation_numbers):
        file_name = dataset.get_sample_file_name()
        new_file_name = "{:0>9d}_{}".format(file_index, name_sub)
        file_index = file_index + 1
#        augmentation_one_image(
#            os.path.join(images_path, "{}.jpg".format(file_name)),
#                        os.path.join(xmls_path, "{}.xml".format(file_name)),
#                        os.path.join(images_path, "{}.jpg".format(new_file_name)),
#                        os.path.join(xmls_path, "{}.xml".format(new_file_name))
#        )
        executor.submit(progress, os.path.join(images_path, "{}.jpg".format(file_name)),
                        os.path.join(xmls_path, "{}.xml".format(file_name)),
                        os.path.join(images_path, "{}.jpg".format(new_file_name)),
                        os.path.join(xmls_path, "{}.xml".format(new_file_name)))



    executor.shutdown(wait=True)










