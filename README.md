# Vit
## Vit implementation in pytorch [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

![vit architecture](https://github.com/markpesic/vit/blob/master/images/vit.png?raw=true)

## Model
```python
from VIT.vit import Vit

model = Vit(
    image_size: (224, 224),
    patch_size: (32, 32),
    num_classes: 100,
    dim: 1024,
    depth: 8,
    hidden_dim: 2048,
    nheads=8,
    dim_head=64,
    dropout=0.1,
    activation='gelu',
    channels=3,
    pool='cls'
)
```

## Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.2010.11929,
  doi = {10.48550/ARXIV.2010.11929},
  url = {https://arxiv.org/abs/2010.11929},
  author = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```