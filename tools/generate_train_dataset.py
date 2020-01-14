import os
import io
import sys
import random
import argparse
import logging

import hashlib

sys.path.append("./")
sys.path.append("./research")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('--ratio', type=float, help='train and test ratio', default=0.95)
    #parser.add_argument('--retrain_weights', type=str, help='weights to restore train path. for example: ./output_dir/weight.h5', default='')
    return parser.parse_args()

def write_to_txt(txt_path, list_name):
    with open(txt_path, "w+") as f:
        for name in list_name:
            f.writelines("{}\n".format(name))

from lxml import etree
import PIL.Image
import tensorflow as tf


from research.object_detection.utils import dataset_util
from research.object_detection.utils import label_map_util

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if data.get('object') != None:
        for obj in data.get('object'):
            difficult_obj.append(int(0))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)

            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            truncated.append(int(0))
            poses.append('Unspecified'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(
        output_filename,
        label_map_dict,
        dataset,
        examples):
    """Creates a TFRecord file from examples.
    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join("data", dataset, 'annotations', example + '.xml')

        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        data['filename'] = "{}.jpg".format(example)

        tf_example = dict_to_tf_example(data, label_map_dict, os.path.join("data", dataset, "images"))
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    dataset = args.dataset
    assert dataset is not None
    shuffle_datset = False

    images_path = os.path.join("data", dataset, "annotations")
    trainval_path = os.path.join("data", dataset, "trainval.txt")
    train_path = os.path.join("data", dataset, "train.txt")
    test_path = os.path.join("data", dataset, "test.txt")

    images_name = os.listdir(images_path)
    images_name = [os.path.splitext(i)[0] for i in images_name]
    if shuffle_datset:
        random.shuffle(images_name)
    print("Top of {} number".format(len(images_name)))

    train_image_list = images_name[:int(args.ratio * len(images_name))]
    test_image_list = images_name[int(args.ratio * len(images_name)):]

    write_to_txt(trainval_path, images_name)
    write_to_txt(train_path, train_image_list)
    write_to_txt(test_path, test_image_list)

    label_map_dict = label_map_util.get_label_map_dict(os.path.join('data', dataset, 'label_map.pbtxt'))
    create_tf_record(os.path.join("data", dataset, "train.record"), label_map_dict, dataset, train_image_list)
    create_tf_record(os.path.join("data", dataset, "val.record"), label_map_dict, dataset, test_image_list)

