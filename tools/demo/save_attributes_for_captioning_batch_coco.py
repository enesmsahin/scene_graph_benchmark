# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import os.path as op
import argparse
import json

from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from detect_utils import detect_objects_on_multiple_images
from visual_utils import draw_bb, draw_rel

import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path

prediction_tsv_files = (
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/coco/coco_caption/ml_decoder.train.label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/coco/coco_caption/ml_decoder.val.label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/coco/coco_caption/ml_decoder.test.label.tsv",
)

batch_size_enes = 4

img_root_dir = "/home/deepuser/deepnas/DISK2/DATASETS/COCO/2014/"

def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1], reverse=True)
        return list(zip(*sorted_dic))
    else:
        return [[], []]


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--img_file", metavar="FILE", help="image path")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the proceed image")
    parser.add_argument("--visualize_attr", action="store_true",
                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    try:
        visual_labelmap = load_labelmap_file(args.labelmap_file)
    except:
        visual_labelmap = None

    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        dataset_attr_labelmap = {
            int(val): key for key, val in
            dataset_allmap['attribute_to_idx'].items()}
    
    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        dataset_relation_labelmap = {
            int(val): key for key, val in
            dataset_allmap['predicate_to_idx'].items()}

    transforms = build_transforms(cfg, is_train=False)

    print(f"Batch Size: {batch_size_enes}")
    for prediction_tsv_file in tqdm(prediction_tsv_files):
        print(f"Starting {prediction_tsv_file}")
        with open(prediction_tsv_file, "r") as pred_tsv_f:
            out_path = op.dirname(prediction_tsv_file)
            out_path = op.join(out_path, "attributes")
            os.makedirs(out_path, exist_ok=True)
            with open(op.join(out_path, op.basename(prediction_tsv_file)), "w") as out_pred_tsv_f: 
                out_line = ""
                lines = pred_tsv_f.readlines()
                for line_batch_idx in tqdm(range(0, len(lines), batch_size_enes)):
                    lines_batch = lines[line_batch_idx:line_batch_idx + batch_size_enes]

                    img_ids = []
                    cv2_imgs = []
                    for line in lines_batch:
                        img_id, curr_preds = line.split("\t")
                        # curr_preds = json.loads(curr_preds)
                        img_path = str(next(Path(img_root_dir).rglob(f"*{img_id}*.jpg")))
                        cv2_img = cv2.imread(img_path)

                        if cv2_img is None:
                            raise Exception(f"Image Not Found: {img_path}")

                        img_ids.append(img_id)
                        cv2_imgs.append(cv2_img)

                    dets_batch = detect_objects_on_multiple_images(model, transforms, cv2_imgs)

                    for cv2_img, img_id, dets in zip(cv2_imgs, img_ids, dets_batch):
                        if len(cv2_img.shape) == 2:
                            img_h, img_w = cv2_img.shape
                        else:
                            img_h, img_w, _ = cv2_img.shape

                        for obj in dets:
                            obj["class"] = dataset_labelmap[obj["class"]]

                        if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
                            for obj in dets:
                                obj["attr"], obj["attr_conf"] = postprocess_attr(dataset_attr_labelmap, obj["attr"], obj["attr_conf"])

                        desired_order = ["class", "conf", "rect", "attr", "attr_conf"]

                        # assert len(desired_order) == len(dets[0])

                        dets = [{key: elem[key] for key in desired_order} for elem in dets]
                        out_dict = {
                            "image_h": img_h,
                            "image_w": img_w,
                            "num_boxes": len(dets),
                            "objects": dets
                        }
                        out_line += img_id + "\t" + json.dumps(out_dict) + "\n" 
                
                out_pred_tsv_f.write(out_line)

        print(f"Finished {prediction_tsv_file}")


if __name__ == "__main__":
    main()
