# BiT-ImageCaptioning-Mult-GPUs

**BiT-ImageCaptioning-Mult-GPUs** is a Python package for generating **Arabic image captions** using **Bidirectional Transformers (BiT)**. This library is designed to provide high-quality and accurate captions for Arabic datasets by leveraging pre-trained deep learning models.


## Installation

Clone the repository:

```bash
git clone https://github.com/Mahmood-Anaam/BiT-ImageCaptioning-Mult-GPUs.git
cd BiT-ImageCaptioning-Mult-GPUs
```

Create  .env for environment variables:

```env
HF_TOKEN = "hugging_face_token"
```

Create  conda environment:

```bash
conda env create -f environment.yml
conda activate sg_benchmark
```

Install Scene Graph Detection for feature extraction:

```bash
cd src\scene_graph_benchmark
python setup.py build develop
```

Download Image captioning model

```bash
cd ..
git lfs install
git clone https://huggingface.co/jontooy/AraBERT32-Flickr8k bit_image_captioning/pretrained_model
```

Install BiT-ImageCaptioning for image captioning:

```bash
cd ..
python setup.py build develop
```



## Quick Start

```python

from bit_image_captioning.feature_extractors.vinvl import VinVLFeatureExtractor
from bit_image_captioning.pipelines.bert_pipeline import BiTImageCaptioningPipeline
from bit_image_captioning.datasets.ok_vqa_dataset import OKVQADataset
from bit_image_captioning.datasets.ok_vqa_dataloader import OKVQADataLoader
from bit_image_captioning.modeling.bert_config import BiTConfig

# Extract image features
feature_extractor = VinVLFeatureExtractor(add_od_labels=BiTConfig.add_od_labels)
# img # (file path, URL, PIL.Image, numpy array, or tensor) 
image_features = feature_extractor([img])
# return List[dict]: List of extracted features for each image.
# [{"boxes","classes","scores","img_feats","od_labels","spatial_features"},]


# Generate a caption

pipeline = BiTImageCaptioningPipeline(BiTConfig)
features,captions = pipeline([img])
print("Generated Caption:", caption)
```





