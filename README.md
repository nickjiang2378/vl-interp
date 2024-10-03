# Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations

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

### Model Weights

The configs for InstructBLIP models are under `src/caption/lavis/configs/`.

In order to get InstructBLIP (7B) working, you should download [these model weights](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) and set the 'pretrained' attribute to their file path in `src/caption/lavis/configs/blip2_instruct_vicuna7b.yaml`.

## Demos

Our paper presents two primary methods to interpret and edit VL representations. The first method creates a model confidence score for model-generated objects by projecting image representations to the language vocabulary and taking a max softmax score of the output probabilities. Our second method target and remove objects from image captions by subtracting the text embeddings of targeted objects from these image representations.

To explore internal model confidences and zero-shot segmentation, check out `demos/internal_confidence.ipynb`.

To erase objects by editing internal representations, run `demos/object_erasure.ipynb`.

## Evals

Generated captions for the hallucination reduction task (Section 5.2) are in `log_results/`. To evaluate CHAIR scores, run
```
python3 metric/chair.py --cap_file <log_file> --cache metric/chair.pkl
```

You may need to run the following in your conda environment before CHAIR works:
```
>>> import nltk
>>> nltk.download('punkt_tab')
```