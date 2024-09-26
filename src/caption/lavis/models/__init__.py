import logging
import torch
from omegaconf import OmegaConf

from src.caption.lavis.models.blip2_models.blip2_vicuna_instruct import (
    Blip2VicunaInstruct,
)
from src.caption.lavis.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from src.caption.lavis.models.blip2_models.blip2_opt import Blip2OPT

from src.caption.lavis.processors.base_processor import BaseProcessor
from src.caption.lavis.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    BlipImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)

LOCAL_MODEL_REGISTRY = {
    "blip2_opt": Blip2OPT,
    "blip2_vicuna_instruct": Blip2VicunaInstruct,
    "blip2_t5_instruct": Blip2T5Instruct,
}

LOCAL_PROCESSOR_REGISTRY = {
    "blip2_image_train": Blip2ImageTrainProcessor,
    "blip_image_train": BlipImageTrainProcessor,
    "blip_image_eval": BlipImageEvalProcessor,
    "blip_caption": BlipCaptionProcessor,
}


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        if cfg.name not in LOCAL_PROCESSOR_REGISTRY:
            raise KeyError("Unknown processor:", cfg.name)
        return (
            # registry.get_processor_class(cfg.name).from_config(cfg)
            LOCAL_PROCESSOR_REGISTRY[cfg.name].from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(
    name, model_type, is_eval=False, device="cpu", vision_only=False
):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    if name not in LOCAL_MODEL_REGISTRY:
        raise KeyError("Unknown model:", name)

    model_cls = LOCAL_MODEL_REGISTRY[name]

    # load model
    model = model_cls.from_pretrained(model_type=model_type, vision_only=vision_only)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors
