# FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

This repository is the official implementation of the [AAAI 2022 paper "FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles"](https://arxiv.org/abs/1911.12199). 

## Requirements

To install requirements:

```setup
conda env create --file environment.yml
```

>📋 This will create a conda environment called tensorflow-py3


## Using FOCUS to generate counterfactual explanations

To train FOCUS for each dataset, run the following commands:

```train
python main.py --sigma=1.0 --temperature=1.0 --distance_weight=0.01 --lr=0.001 --opt=adam --model_name=<MODEL_NAME> --data_name=<DATA NAME> --model_type=<MODEL TYPE> --distance_function=<euclidean/cosine/etc>
```

>📋  This will create another folder in the main directory called 'results', where the results files will be stored.


## Pre-trained Models

The pretrained models are available in the models folder
