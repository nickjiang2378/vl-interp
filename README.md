# Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations

## Code structure

Main files:
```
src/
  caption/
    __init__.py     # Imports available captioning models
    base_engine.py  # Base class for captioning models
    instruct_blip_engine.py  # InstructBLIP captioning models; extends from base.py
    lavis/       # Source code for Instruct-BLIP models; note that instruct-blip-{7b,13b} use lavis/models/blip2_models/blip2_vicuna_instruct.py)
    llava/       # Source code for LLaVA models
```

The configs for InstructBLIP models are under `src/caption/lavis/configs/`.


## Setup

### Files
```
git clone git@github.com:nickjiang2378/vl-interp.git
cd vl-interp
```

### Environment

```
# Create a new conda environment
conda create -n vl python=3.9
conda activate vl

# Set up LLaVA repo
mkdir src/caption/llava
cd src/caption/llava
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip3 install -e .

# cd back into repo root
cd ../../../../
pip3 install -e .

# Install some remaining packages
pip3 install lightning openai-clip transformers==4.37.2
```

## Demos

To explore internal model confidences and zero-shot segmentation, check out `demos/internal_confidence.ipynb`.

## Evals

