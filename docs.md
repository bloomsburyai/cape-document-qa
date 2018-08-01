# Cape Document QA

This repo contains a machine reading model, built on top of on document-qa, a powerful machine reading library and 
recent state of the art approach in open domain question answering. 

### Who is this repo for:

This repo enables training and evaluation of models, and to provide a standard model for Cape.
the primary purpose of this repo is to train and evalauate models that implement the
 `cape-machine-reader.cape_machine_reader_model.CapeMachineReaderModelInterface` interface. 

### Who is this repo not for:

This repo is not designed to be used "as is" in a production environment. The model functionality is kept deliberately minimal.
Please use `cape-responder`, or `cape-webservices` to use models for downstream tasks.

## About the models: 

cape-document-qa allows users to train and evaluate their own machine reading models, and allows the 
simultaneous training of both supervised and semi-supervised machine reading tasks, such as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
 and [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) 

The original document-qa library can be found here: https://github.com/allenai/document-qa (We link to it as a submodule).
The original publication describing some of how document-qa works can be found here: 
[Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/abs/1710.10723).

Cape-document-qa models are similar to the "shared-norm" model described in the paper above, but differ in some details,
the biggest being that we include an ELMO language model in our models, and we co-train models on both Squad and
TriviaQA, retaining strong performance on both datasets.

In order to have clean installs, and to enable functionality such as tensorflow 1.7, we have made several patches to document-qa. 
These patches have been kept as minimal as possible, but do bear this in mind when using the codebase.

## Setup

`Cape-document-qa` can be used simply to load pretrained models and use them in inference mode, or can also be used
to train models from scratch or finetune models.

### Install:

Training and running cape-document-qa models requires tensorflow 1.7 You should install tensorflow using the documentation
especially if using gpus with CUDA and CUDNN

To install as a site-package:
```
pip install --upgrade --process-dependency-links git+https://github.com/bloomsburyai/cape-document-qa
```

To use locally, and run commands from the project root (recommended for users planning on training their own models):

```
git clone https://github.com/bloogram/cape-document-qa.git
git submodule init
git submodule update
cd cape-document-qa
export PYTHONPATH="${PYTHONPATH}:./document-qa/"
pip install --upgrade --process-dependency-links git+https://github.com/bloomsburyai/cape-machine-reader
pip install -r requirements.txt
```

You will also need a model. A Pretrained model will be downloaded automatically when the library is first imported.
This model will be downloaded to `cape_document_qa/storage/models` and contains all required data out of the box.
The space footprint of this model is about 5GB, the majority of which are Glove Word vectors.

To check that the install is successful, we can run the tests:

```
pytest cape_document_qa
```


### Setup for users wanting to train models

If you are training models, you may find it easier to do a local install.
In this case, you should ensure that the docqa module within document-qa is on your PYTHONPATH.

## Training Models

Training your own models is encouraged. You can use the `cape-document-qa` training scripts to train and tweak
models. If you want to define your own architecture, or even use your own codebase to train a model, this should
be achievable too, you just need to make your model inherit the 
cape-machine-reader.cape_machine_reader_model.CapeMachineReaderModelInterface interface. 
(see cape_document_qa.cape_docqa_machine_reader for an example).

We suggest using our model as a good starting place to fine tune your own models. 

Training a model requires you to

1) download the data needed to train models
2) preprocess the data
3) run the training script
4) evaluate the model
5) make the model "production ready".

These steps are described below. Each is can be achieved by running one or two scripts

### Data:

Datasets for training, and other resources (including ELMO parameters) are downloaded and handled
by the `cape_document_qa.training.download_training_resources` script:

```
python -m cape_document_qa.training.download_training_resources
```
Training data will be automatically triggered when `cape_document_qa.training` or `cape_document_qa.preprocess` is imported

This script will download Elmo parameters, the squad dataset, the web and wiki triviaqa datasets and glove vectors

By default, we expect source data to be stored in `\~/data` and preprocessed data to be
stored in `{project_root}/data`. These can be changed by altering `cape_document_qa/cape_config.py`


### Preprocessing:

Before training, There is significant preprocessing that needs to be done.
This process can take several hours (if preprocessing all of squad and triviaqa). By default
most of the pipeline is multi-processed (default is 8 processes)

Preprocessing can be run by:
```
python -m cape_document_qa.cape_preprocess --dataset_dict path/to/datsets_dict.json
```

You'll need to define a `dataset_dict.json`. This simply tells the preprocessing what data should
go into what data fold. The default `dataset_dict` is shown below:
```
{
    "triviaqa_datasets": ["wiki", "web"],
    "squad_datasets": {
        "squad": {
            "train": ["squad-train-v1.1.json"],
            "dev": ["squad-dev-v1.1.json"],
            "test": ["squad-dev-v1.1.json"],
        }
    }
}
```
(you can run without specifying a `dataset_dict`, which will preprocess triviqaQA Wiki, triviaQA Web and Squad)

Preprocessing will perform the following steps (in order):

* Tokenize squad documents
* Tokenize triviaQA documents
* Tokenize questions and build supporting objects, and pickle them
* Create elmo token embeddings for the whole dataset's vocab


### Adding your own datasets:

adding your own datasets should be straightforward. Save them using the squad-v1.1 format, place them
in the same location as the squad json files (specified by cape_document_qa.cape_config.SQUAD_SOURCE_DIR)
and then preprocess
them as you did for squad, by running `cape_document_qa.cape_preprocess` with an updated `dataset_dict.json`, e.g.:
```
{
    "triviaqa_datasets": ["wiki", "web"],
    "squad_datasets":{
        "squad": {
            "train": ["squad-train-v1.1.json"],
            "dev": ["squad-dev-v1.1.json"],
            "test": ["squad-dev-v1.1.json"],
        },
        "my_dataset" : {
            "train": ["my_dataset-train-v1.1.json"]
            "dev": ["my_dataset-dev-v1.1.json"],
            "test": ["my_dataset-test-v1.1.json"],
        }
    }
}
```


### Training a model from scratch:

Once all the data has been preprocessed, you can train a model. This requires a gpu, we recommend
at least 12GB of GPU memory.
Training can be slow, especially when co-training triviaqa and Squad (over 50 hours to converge on some
slower hardware), so be aware.
Squad-only models will train much faster (10-15 hours)

To train a model, use `cape_document_qa.training.cape_ablate`

to specify what datasets to train your model on, you can define a dataset_sampling_dict, e.g.
to train on triviaqa web, triviaqa wiki, squad and a 2x oversampling of `my_dataset`, the
`dataset_sampling_dict.json` would look like this:

```
{
      "wiki": 1,
      "web": 1,
      "squad": 1,
      "my_dataset": 2,
}
``` 

Training a model could the be done using:

```
python -m cape_document_qa.training.cape_ablate name_of_my_model --dataset_sampling path/to/my_datset_sampling.json --cudnn
```

After some preparatory preprocessing and loading (sometimes up to 1 hour if training on a lot of data),
a model will start to train. It will create a model directory, and you can track training progress using tensorboard
and pointing it at the logs subdirectory of the model directory.

```
tensorboard --logdir /path/to/my/model/logs/.
```


### Resume a training run:

Sometimes runs break down, or you may want to try to fine tune a pretrained model with new data.
You can resume training a model using `cape_document_qa.training.cape_resume_training.py`:

```
python -m cape_document_qa.training.cape_resume_training path/to/my/model --dataset_sampling path/to/my_datset_sampling.json
```


### Evaluating a model:

When a model has finished, there are two evalation scripts. One uses document-qa's own evaluation
pipeline and can be called like:

```
python -m cape_document_qa.evaluation.cape_docqa_eval path/to/my/model \
    --paragraph_output path/to/paragraph_output.csv \
    --aggregated_output path/to/aggregated_output.csv \
    --official_output path/to/official_output.json \
    --datasets squad wiki web my_dataset
```
This script will produce three files, one with paragraph level answers, one with EM and F1 scores for 
each dataset, and an "official" json format, like that used for squad model evaluation.

The other evaluation uses Cape's answer generator, which enables faster generation of the top k answers
as well as several heuristics that are useful for the user experience

This can be called like:
```
python -m cape_document_qa.evaluation.cape_multidoc_eval path/to/my/model -k 5 --datasets squad wiki web my_dataset
```
which will produce a file for each dataset with the top k answers for each question. 

### Making a model "production ready":

Once you have trained a model and you are happy with the results, you can gather the resources and slim down
the model files using 

```
cape_document_qa.cape_productionize_model --target_model path/to/my/trained_model --output_dir path/to/my/output_model
```
This will also convert the RNNs to be CPU compatible, so can be run on systems without nvidia gpus. This
production ready model can now be used by `cape_document_qa.cape_docqa_machine_reader` and be used
by the rest of the stack.
