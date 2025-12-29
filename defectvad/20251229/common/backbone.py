# common/bacbone.py

import os


BACKBONE_WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "wide_resnet50_2-32ee1156.pth",
    "efficientnet_b5": "efficientnet_b5_lukemelas-1a07897c.pth",
    
    # https://huggingface.co/timm/wide_resnet50_2.tv_in1k/tree/main
    "wide_resnet50_2.tv_in1k": "wide_resnet50_2.tv_in1k",
    "resnet50.tv_in1k": "resnet50.tv_in1k",

    # https://huggingface.co/zgcr654321/pretrained_models/tree/main/dinov2_pretrain_official_pytorch_weights
    "dinov2_vit_small_14": "dinov2_vits14_pretrain.pth",
    "dinov2_vit_base_14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vit_large_14": "dinov2_vitl14_pretrain.pth",
    "dinov2reg_vit_small_14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2reg_vit_base_14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2reg_vit_large_14": "dinov2_vitl14_reg4_pretrain.pth",

    # https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/tree/main
    # https://huggingface.co/timm/cait_s24_224.fb_dist_in1k/tree/main
    # https://huggingface.co/timm/deit_base_distilled_patch16_224.fb_in1k/tree/main
    # https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/tree/main
    "cait_s24_224": "cait_s24_224.fb_dist_in1k",
    "cait_m48_448": "cait_m48_448.fb_dist_in1k",
    "deit_base_distilled_patch16_224": "deit_base_distilled_patch16_224.fb_in1k",
    "deit_base_distilled_patch16_384": "deit_base_distilled_patch16_384.fb_in1k",
}


def get_backbone_path(backbone: str):
    backbone_dir = os.getenv('BACKBONE_DIR')
    
    if backbone_dir is None:
        raise RuntimeError(
            "[ERROR] BACKBONE_DIR not set in environment variables."
        )

    if backbone.startswith("cait"):
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.fb_dist_in1k")
        weights_path = os.path.join(backbone_dir, dirname, "model.safetensors")
    elif backbone.startswith("deit"):
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.fb_in1k")
        weights_path = os.path.join(backbone_dir, dirname, "model.safetensors")
    elif backbone in ("resnet50.tv_in1k", "wide_resnet50_2.tv_in1k"):
        dirname = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.tv_in1k")
        weights_path = os.path.join(backbone_dir, dirname, "model.safetensors")
    else:
        filename = BACKBONE_WEIGHT_FILES.get(backbone, f"{backbone}.pth")
        weights_path = os.path.join(backbone_dir, filename)

    if os.path.isfile(weights_path):
        print(f" > {backbone} weight is loaded from {os.path.basename(weights_path)}.")
    else:
        print(f" > {backbone} weight not found in {os.path.basename(weights_path)}.")
    return weights_path
