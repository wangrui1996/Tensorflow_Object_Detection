import argparse
from libs.protobuf_generator import generate_ssd_model
from object_detection.protos import pipeline_pb2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('--ratio', type=float, help='train and test ratio', default=0.95)
    #parser.add_argument('--retrain_weights', type=str, help='weights to restore train path. for example: ./output_dir/weight.h5', default='')
    return parser.parse_args()





if __name__ == '__main__':
    args = parse_args()
    train_eval_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_eval_config.model.ssd.CopyFrom(generate_ssd_model(1))
    print(train_eval_config.model)


