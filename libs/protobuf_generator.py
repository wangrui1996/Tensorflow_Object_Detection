from object_detection.protos import eval_pb2
from object_detection.protos import graph_rewriter_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2


from object_detection.protos import ssd_pb2

def generate_ssd_model(num_classes):

    ssd_config = ssd_pb2.Ssd()
    ssd_config.num_classes = num_classes

    # config box_coder
    from object_detection.protos import box_coder_pb2
    from object_detection.protos import faster_rcnn_box_coder_pb2
    box_coder = box_coder_pb2.BoxCoder()
    faster_rcnn_box_coder = faster_rcnn_box_coder_pb2.FasterRcnnBoxCoder()
    faster_rcnn_box_coder.y_scale = 10.0
    faster_rcnn_box_coder.x_scale = 10.0
    faster_rcnn_box_coder.height_scale = 5.0
    faster_rcnn_box_coder.width_scale = 5.0
    box_coder.faster_rcnn_box_coder.CopyFrom(faster_rcnn_box_coder)
    ssd_config.box_coder.CopyFrom(box_coder)

    # config matcher
    from object_detection.protos import matcher_pb2
    from object_detection.protos import argmax_matcher_pb2
    argmax_matcher = argmax_matcher_pb2.ArgMaxMatcher()
    argmax_matcher.matched_threshold = 0.5
    argmax_matcher.unmatched_threshold = 0.5
    argmax_matcher.ignore_thresholds = False
    argmax_matcher.negatives_lower_than_unmatched = True
    argmax_matcher.force_match_for_each_row = True
    matcher = matcher_pb2.Matcher()
    matcher.argmax_matcher.CopyFrom(argmax_matcher)
    ssd_config.matcher.CopyFrom(matcher)

    # config anchor generator
    from object_detection.protos import anchor_generator_pb2
    from object_detection.protos import ssd_anchor_generator_pb2
    ssd_anchor_generator = ssd_anchor_generator_pb2.SsdAnchorGenerator()
    ssd_anchor_generator.num_layers = 6
    ssd_anchor_generator.min_scale = 0.2
    ssd_anchor_generator.max_scale = 0.95
    ssd_anchor_generator.aspect_ratios.extend([1.0, 2.0, 0.5, 3.0, 0.3333])
    anchor_generator = anchor_generator_pb2.AnchorGenerator()
    anchor_generator.ssd_anchor_generator.CopyFrom(ssd_anchor_generator)
    ssd_config.anchor_generator.CopyFrom(anchor_generator)

    return ssd_config

def generate_train_config():
    pass

def generate_train_input_reader():
    pass

def generate_eval_config():
    pass

def generate_eval_input_reader():
    pass

