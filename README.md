# cape-document-qa [![CircleCI](https://circleci.com/gh/bloomsburyai/cape-document-qa.svg?style=svg&circle-token=d7f735b32660655a8b56495fffdb8726eaa56594)](https://circleci.com/gh/bloomsburyai/cape-document-qa)
This repo contains a machine reading model, built on top of on document-qa, a powerful machine reading library and  recent state of the art approach in open domain question answering. 

More detailed tutorial/documentation can be found [Here](docs.md)

### Who is this repo for:

This repo enables training and evaluation of models, and to provide a reference model for Cape.
the primary purpose of this repo is to train and evaluate models that implement the
 `cape-machine-reader.cape_machine_reader_model.CapeMachineReaderModelInterface` interface. 

### Who is this repo not for:

This repo is not designed to be used "as is" in a production environment. The model functionality is kept deliberately minimal.
Please use `cape-responder`, or `cape-webservices` to use models for downstream tasks.
