import cv2
import sys
import numpy as np
import xml.etree.ElementTree as ET
import random

def get_max_for_list(data_list):
    max_data = -1
    for data in data_list:
        if max_data < data:
            max_data = data
    return max_data

def get_min_for_list(data_list):
    min_data = sys.maxsize
    for data in data_list:
        if min_data > data:
            min_data = data
    return min_data


def get_axis_from_object(object):
    return int(object.find("bndbox").find("xmin").text), \
    int(object.find("bndbox").find("ymin").text), \
    int(object.find("bndbox").find("xmax").text), \
    int(object.find("bndbox").find("ymax").text)

def set_axis_in_object(object, xmin, ymin, xmax, ymax, height, width):
    object.find("bndbox").find("xmin").text = str(max(1, int(xmin)))
    object.find("bndbox").find("ymin").text = str(max(1, int(ymin)))
    object.find("bndbox").find("xmax").text = str(min(width, int(xmax)))
    object.find("bndbox").find("ymax").text = str(min(height, int(ymax)))


def get_objects_restrict(objects):
    xmin, ymin, xmax, ymax = sys.maxsize, sys.maxsize, -1, -1
    for object in objects:
        xin, yin, xax, yax = get_axis_from_object(object)
        if xin < xmin:
            xmin = xin
        if yin < ymin:
            ymin = yin
        if xax > xmax:
            xmax = xax
        if yax > ymax:
            ymax = yax
    return xmin, ymin, xmax, ymax

# 平移和旋转有区别，
# 旋转需要通过四个点判断最后的框
# 平移两个点即可
def alter_axis_by_M(objects, M, height, width, is_translation=True):
    x_M = M[0]
    y_M = M[1]
    for object in objects:
        xmin, ymin, xmax, ymax = get_axis_from_object(object)
        n_xmin = x_M[0] * xmin + x_M[1] * ymin + x_M[2]
        n_ymin = y_M[0] * xmin + y_M[1] * ymin + y_M[2]
        n_xmax = x_M[0] * xmax + x_M[1] * ymax + x_M[2]
        n_ymax = y_M[0] * xmax + y_M[1] * ymax + y_M[2]
        if is_translation:
            set_axis_in_object(object, n_xmin, n_ymin, n_xmax, n_ymax, height, width)
            return
        left_x, left_y, right_x, right_y = xmin, ymax, xmax, ymin
        n_left_x = x_M[0] * left_x + x_M[1] * left_y + x_M[2]
        n_left_y = y_M[0] * left_x + y_M[1] * left_y + y_M[2]
        n_right_x = x_M[0] * right_x + x_M[1] * right_y + x_M[2]
        n_right_y = y_M[0] * right_x + y_M[1] * right_y + y_M[2]
        set_axis_in_object(object, get_min_for_list([n_xmin, n_xmax, n_left_x, n_right_x]), get_min_for_list([n_ymin, n_ymax, n_left_y, n_right_y]),
                           get_max_for_list([n_xmin, n_xmax, n_left_x, n_right_x]), get_max_for_list([n_ymin, n_ymax, n_left_y, n_right_y]), height, width)




# flip image in horizeontal
def flip_horizeontal_image(image, xml_root, prob=0.9):
    if random.random() > prob:
        return image, xml_root
    h, w, _ = image.shape
    image = cv2.flip(image, 1)
    objects = xml_root.findall("object")
    for object in objects:
        xmin, ymin, xmax, ymax = get_axis_from_object(object)
        set_axis_in_object(object, w - xmin, ymin, w - xmax, ymax, h, w)
    return image, xml_root

# translation image
def translation_image(image, xml_root, prob=0.9):
    if random.random() > prob:
        return image, xml_root
    h, w, _ = image.shape
    objects = xml_root.findall("object")
    obs_xmin, obs_ymin, obs_xmax, obs_ymax = get_objects_restrict(objects)
    x_offset = random.randint(-int(obs_xmin * 0.1), int(0.1*(w - obs_xmax)))
    y_offset = random.randint(-int(obs_ymin *0.1), int(0.1*(h - obs_ymax)))
    axis_x_alter = [1, 0, float(x_offset)]
    axis_y_alter = [0, 1, float(y_offset)]
    M = np.float32([axis_x_alter, axis_y_alter])
    image = cv2.warpAffine(image, M, (w, h))
    alter_axis_by_M(objects, M, h, w)
    return image, xml_root


def rotation_image(image, xml_root, angle_range = (-10, 10), scale = 1, prob=0.9):
    if random.random() > prob:
        return image, xml_root
    h, w, _ = image.shape
    M = cv2.getRotationMatrix2D((random.randint(0, int(1.05*w)), random.randint(-h//10, int(1.05*h))), random.randint(angle_range[0], angle_range[1]), scale)
    image = cv2.warpAffine(image, M, (w, h))
    objects = xml_root.findall("object")
    alter_axis_by_M(objects, M, h, w, False)
    return image, xml_root

def gaussianblur_image(image, prob=0.5):
    if random.random() > prob:
        return image

    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    return image

def check_object_position_in_image(root_xml, height_restrict = 30, width_restrict =30):
    objects = root_xml.findall("object")
    for object in objects:
        xmin, ymin, xmax, ymax = get_axis_from_object(object)
        if xmax - xmin < width_restrict or ymax - ymin < height_restrict:
            return False
    return True

def is_have_object_in_xml(xml_path):
    xml = ET.parse(xml_path)
    root = xml.getroot()
    if root.find("object"):
        return True
    return False

def augmentation_one_image(src_image_path, src_xml_path, dst_image_path, dst_xml_path, try_restrict_numbers = 6):
#    print(src_image_path, src_xml_path, dst_image_path, dst_xml_path)
    image = cv2.imread(src_image_path)
    xml = ET.parse(src_xml_path)
    root = xml.getroot()
    try_number = 0
    generate_ok = False
    while try_number < try_restrict_numbers:
        image, xml = rotation_image(image, root)
        image, xml = translation_image(image, root)
        image, xml = flip_horizeontal_image(image, root)
        if check_object_position_in_image(root):
            image = gaussianblur_image(image)
            generate_ok = True
            break
        try_number = try_number + 1
    if generate_ok:
        cv2.imwrite(dst_image_path, image)
        tree = ET.ElementTree(xml)
        tree.write(dst_xml_path, encoding="utf-8", xml_declaration=True)




if __name__ == '__main__':
    src_image_path = "/home/rui/dataset/torn/test/demo.jpg"
    src_xml_path = "/home/rui/dataset/torn/test/demo.xml"
    dst_image_path = "/home/rui/dataset/torn/test/result.jpg"
    dst_xml_path = "/home/rui/dataset/torn/test/result.xml"
    img = cv2.imread(src_image_path)
    xml = ET.parse(src_xml_path)
    root = xml.getroot()
    #img, xml = translation_image(img, root)
    #img, xml = flip_horizeontal_image(img, root)
    img, xml = rotation_image(img, root)
    if not check_object_position_in_image(root):
        print("Error")
    img = gaussianblur_image(img)
    tree = ET.ElementTree(xml)
    tree.write(dst_xml_path, encoding="utf-8", xml_declaration=True)
    cv2.imwrite(dst_image_path, img)


