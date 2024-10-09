# Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion (ECCV24)
This is the official code for our paper: <a href='https://arxiv.org/pdf/2407.14143'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

The code is currently being collated, and configs of other datasets along with more readable code will be released at a later time.
## Getting Started

## Environment
create enviroment using Miniconda (or Anaconda)
```
conda create -n continual_clip python=3.8
conda activate continual_clip
```
install dependencies:
```bash
bash setup_environment.sh
``` 
### Running scripts

We provide the scripts for imagenet100. Please run:

```
python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="[imagenet1k_path]" \
    class_order="class_orders/imagenet100.yaml"
```
The dataset_root folder should contain the train and val folders.
```
imagenet1k_path
├── train
│   ├── n01440764 
│   └── ···
├── val
│   ├── n01440764 
│   └── ···
```


### datasets
Other datasets and configs will be released soon


## Acknowledgement
Our method implementation is based on the [Continual-CLIP](https://github.com/vgthengane/Continual-CLIP).

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@misc{huang2024rapf,
      title={Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion}, 
      author={Linlan Huang and Xusheng Cao and Haori Lu and Xialei Liu},
      year={2024},
      eprint={2407.14143},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14143}, 
}
```

## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact <a href="huanglinlan@mail.nankai.edu.cn">huanglinlan@mail.nankai.edu.cn</a> 
