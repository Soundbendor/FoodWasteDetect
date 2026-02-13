import os
from copy import deeepcopy

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch

from util.util import parse_args, parse_cfg


def train_yoloe():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_root = cfg["paths"]["yoloe_grounding_root"]

    # Option 1: Use Python dictionary
    # these datasets should auto-download
    data = dict(
        train=dict(
            yolo_data=[os.path.join(ds_root, "lvis.yaml")],
            # INFO: can we get away without any grounding data?
        ),
        val=dict(yolo_data=["lvis.yaml"]),
    )

    # Option 2: Use YAML file (yoloe_data.yaml)
    # train:
    #   yolo_data:
    #     - Objects365.yaml
    #   grounding_data:
    #     - img_path: flickr/full_images/
    #       json_file: flickr/annotations/final_flickr_separateGT_train_segm.json
    #     - img_path: mixed_grounding/gqa/images
    #       json_file: mixed_grounding/annotations/final_mixed_train_no_coco_segm.json
    # val:
    #   yolo_data:
    #     - lvis.yaml

    model = YOLOE("yoloe-26x-seg.pt")

    # freeze all layers
    model.eval

    # unfreeze select weights
    model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
    model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
    model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

    if getattr(model.model[-1], "one2one_cv3", None) is not None:
        model.model[-1].one2one_cv3[0][2] = deepcopy(
            model.model[-1].cv3[0][2]
        ).requires_grad_(True)
        model.model[-1].one2one_cv3[1][2] = deepcopy(
            model.model[-1].cv3[1][2]
        ).requires_grad_(True)
        model.model[-1].one2one_cv3[2][2] = deepcopy(
            model.model[-1].cv3[2][2]
        ).requires_grad_(True)

    model.train(
        data=data,  # or data="yoloe_data.yaml" if using YAML file
        batch=128,
        epochs=30,
        close_mosaic=2,
        optimizer="AdamW",
        lr0=2e-3,
        warmup_bias_lr=0.0,
        weight_decay=0.025,
        momentum=0.9,
        workers=4,
        trainer=YOLOESegTrainerFromScratch,
        device="0,1,2,3,4,5,6,7",
    )


if __name__ == "__main__":
    train_yoloe()
