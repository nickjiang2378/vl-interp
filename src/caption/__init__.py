from typing import Dict, Type

from .base import CaptionEngine  # noqa: F401
from .ofa_engine import OFACaptionEngine
from .blip_engine import (
    BLIP2COCOBase,
    BLIP2Base,
    BLIP2COCOLarge,
    BLIP2Large,
    BLIP2COCOT5Large,
    BLIP2T5Large,
)
from .entropy_threshold import (
    EntropyThresholdBLIP2Base,
    EntropyThresholdBLIP2COCOBase,
    EntropyThresholdInstructBLIPEngine,
)
from .instruct_blip_engine import (
    InstructBLIPVicuna7B,
    InstructBLIPVicuna13B,
    InstructBLIPFlanT5XL,
    InstructBLIPFlanT5XXL,
)
from .llava_engine import LLaVA7B, LLaVA13B

CAPTION_ENGINES: Dict[str, Type[CaptionEngine]] = {
    "OFA (Large + Caption)": OFACaptionEngine,
    "BLIP2 (OPT, COCO, 6.7B)": BLIP2COCOLarge,
    "BLIP2 (OPT, COCO, 2.7B)": BLIP2COCOBase,
    "BLIP2 (T5, COCO, flanT5XL)": BLIP2COCOT5Large,
    "BLIP2 (OPT, 6.7B)": BLIP2Large,
    "BLIP2 (OPT, 2.7B)": BLIP2Base,
    "BLIP2 (T5, flanT5XL)": BLIP2T5Large,
}

CAPTION_ENGINES_CLI: Dict[str, Type[CaptionEngine]] = {
    "ofa": OFACaptionEngine,
    "blip2-coco": BLIP2COCOLarge,
    "blip2-base-coco": BLIP2COCOBase,
    "blip2-t5-coco": BLIP2COCOT5Large,
    "blip2": BLIP2Large,
    "blip2-base": BLIP2Base,
    "blip2-t5": BLIP2T5Large,
    "ent-blip2-base-coco": EntropyThresholdBLIP2COCOBase,
    "ent-blip2-base": EntropyThresholdBLIP2Base,
    "instruct-blip-7b": InstructBLIPVicuna7B,
    "instruct-blip-13b": InstructBLIPVicuna13B,
    "instruct-blip-flanxl": InstructBLIPFlanT5XL,
    "instruct-blip-flanxxl": InstructBLIPFlanT5XXL,
    "ent-instruct-blip": EntropyThresholdInstructBLIPEngine,
    "llava-7b": LLaVA7B,
    "llava-13b": LLaVA13B,
}
