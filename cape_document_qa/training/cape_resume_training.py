# Copyright 2018 BLEMUNDSBURY AI LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from docqa.trainer import _train_async
from docqa.model_dir import ModelDir
import tensorflow as tf
from cape_document_qa.cape_config import TrainConfig
from cape_document_qa.training.cape_ablate import prepare_data
from typing import Optional, Dict


def resume_training(model_to_resume: str,
                    dataset_oversampling: Dict[str, int],
                    checkpoint: Optional[str] = None,
                    epochs: Optional[int] = None
                    ):
    """Resume training on a partially trained model (or finetune an existing model)


    :param model_to_resume: path to the model directory of the model to resume training
    :param dataset_oversampling: dictionary mapping dataset names to integer counts of how much
       to oversample them
    :param checkpoint: optional string to specify which checkpoint to resume from. Uses the latest
         if not specified
    :param epochs: Optional int specifying how many epochs to train for. If not detailed, runs for 24
    """
    out = ModelDir(model_to_resume)
    train_params = out.get_last_train_params()
    evaluators = train_params["evaluators"]
    params = train_params["train_params"]
    params.num_epochs = epochs if epochs is not None else 24
    model = out.get_model()

    notes = None
    dry_run = False
    data = prepare_data(model, TrainConfig(), dataset_oversampling)
    if checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(out.save_dir)

    _train_async(
        model=model,
        data=data,
        checkpoint=checkpoint,
        parameter_checkpoint=None,
        save_start=False,
        train_params=params,
        evaluators=evaluators,
        out=out,
        notes=notes,
        dry_run=dry_run,
        start_eval=False
    )


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name', help='name of model to examine')
    parser.add_argument('-s', '--dataset_sampling', dest='dataset_sampling'
                        , default='', help='path to sampling_dict.json (see readme for more info')
    args = parser.parse_args()
    if args.dataset_sampling == '':
        dataset_sampling = {'wiki': 1, 'web': 1, 'squad': 1}
    else:
        dataset_sampling = json.load(open(args.dataset_oversampling))
    resume_training(args.model, dataset_sampling)


if __name__ == "__main__":
    main()
