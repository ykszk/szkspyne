# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import os
import time
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import torch

# from predictor import VisualizationDemo
from szkspyne.dtron2 import utils
from szkspyne.labelme_utils import load_ndjson


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; " "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        required=True,
    )
    parser.add_argument(
        "--metadata",
        help="Load metadata json file",
    )
    parser.add_argument("--ndjson", help='Save one ndjson instaed of json files')
    parser.add_argument("--coco", help='Save results in COCO format (json)')
    parser.add_argument("--verbose", help="Verbose mode", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    dataset_name = 'scoliosis_val'
    metadata = dict(
        thing_classes=["vertebra"],
        keypoint_names=['TL', 'TR', 'BL', 'BR'],
        keypoint_flip_map=[('TL', 'TR'), ('BL', 'BR')],
    )
    if args.metadata:
        with open(args.metadata) as f:
            metadata = json.load(f)

    from detectron2.data import MetadataCatalog

    MetadataCatalog.get(dataset_name).set(**metadata)

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    if len(args.input) == 1 and args.input[0].endswith('.ndjson'):
        args.input = [e['file_name'] for e in load_ndjson(args.input[0])]
    ndjson = []
    coco_annotation = []
    coco_images = []
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions = predictor(img)
        if args.verbose:
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    (
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished"
                    ),
                    time.time() - start_time,
                )
            )

        instances = predictions["instances"].to(torch.device("cpu"))
        visualizer = Visualizer(img, MetadataCatalog.get(dataset_name), instance_mode=ColorMode.IMAGE)
        visualized_output = visualizer.draw_instance_predictions(predictions=instances)

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output
        visualized_output.save(out_filename)

        filename = os.path.splitext(os.path.basename(path))[0] + '.json'
        if args.ndjson:
            predictions['filename'] = filename
            ndjson.append(json.dumps(predictions, cls=utils.PredictionEncoder, separators=(',', ':')))
        else:
            with open(os.path.join(args.output, filename), 'w') as f:
                json.dump(predictions, f, cls=utils.PredictionEncoder, indent=2)
        if args.coco:
            image_id = len(coco_images)
            coco_image = dict(
                file_name=filename,
                height=img.shape[0],
                width=img.shape[1],
                id=image_id,
            )
            coco_images.append(coco_image)
            result_coco = instances_to_coco_json(instances, image_id)
            coco_annotation.extend(result_coco)
    if args.ndjson:
        with open(args.ndjson, 'w') as f:
            for line in ndjson:
                f.write(line)
                f.write('\n')
    if args.coco:
        info = dict(
            description='detectron2 prediction',
            data_created=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        categories = [{"id": i, "name": c} for i, c in enumerate(metadata['thing_classes'])]
        coco = dict(
            info=info,
            images=coco_images,
            annotations=coco_annotation,
            categories=categories,
        )
        with open(args.coco, 'w') as f:
            json.dump(coco, f)
