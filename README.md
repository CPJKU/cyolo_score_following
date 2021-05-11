## Multi-modal Conditional Bounding Box Regression for Music Score Following

This repository contains the corresponding code for our paper

>[Henkel F.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/florian-henkel/) and 
>[Widmer G.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/) <br>
"[Multi-modal Conditional Bounding Box Regression for Music Score Following](https://arxiv.org/pdf/2105.04309.pdf)".<br>
*In Proceedings of the 29th European Signal Processing Conference (EUSIPCO)*, 2021

### Data
The data used in this paper can be found [*here*](https://zenodo.org/record/4745838/files/msmd.zip?download=1) 
and should be placed in ```cyolo_score_following/data```. If you install the package
properly (see instructions below) this will be done automatically for you.

### Videos
In the folder [`videos`](https://github.com/CPJKU/cyolo_score_following/tree/eusipco-2021/videos) 
you will find several pieces from the test set, where our best performing model follows an incoming musical performance.

## Getting Started
If you want to try our code, please follow the instructions below.

### Setup and Requirements

First, clone the project from GitHub:

`git clone https://github.com/CPJKU/cyolo_score_following.git`

Move to the cloned folder:

`cd cyolo_score_following`

In the cloned folder you will find an anaconda environment file which you should install using the following command:

`conda env create -f environment.yml`

Activate the environment:

`conda activate cyolo_score_following`

Finally, install the project in the activated environment:

`python setup.py develop --user`

This last command will also download and extract the data to `cyolo_score_following/data`

### Check if everything works

To verify that everything is correctly set up, run the following command:

 ```python test.py --test_dir ../data/msmd/msmd_test --test_piece Anonymous__lanative__lanative_synth --plot --gt_only```
 
This will visualize the ground truth for the given test piece.
(Note: The `--plot` mode does not support audio playback. If you want audio, you have to create a video which will be explained below.)

## Training

If you want to train your own models, you will need to run `train.py`. This script takes several parameters
to specify the network architecture and the training procedure. The most important parameters that you will need to set are
the paths to the train and validation set, the model configuration file (see `cyolo_score_following/models/configs`) 
and whether you want to turn on data augmentation or not.
You can also provide a log and a dump directory where the statistics during training and validation as well as the model parameters will be stored. 
The logs can be visualized by using [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html).

To give you an example, if you want to train the model called *CYOLO* in Table 2, run the following command 
(please set the correct paths for logging and dumping the model, and choose a `--tag` for the experiment):

`python train.py --train_set ../data/msmd/msmd_train --val_set ../data/msmd/msmd_valid 
--config ./models/configs/cyolo.yaml --augment --dump_root <DUMP-DIR> --log_root <LOG-DIR> --tag cyolo`

If you have enough memory you can additionally specify `--load_wav`, which will load the audio wav files into the memory.

It is also possible to train a model using multiple GPUs, e.g., if you have access to four GPUs you can train the previous model by calling:
`python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=<IP>
--master_port=<PORT> --use_env train.py --train_set ../data/msmd/msmd_train --val_set ../data/msmd/msmd_valid 
--config ./models/configs/cyolo.yaml --augment --dump_root <DUMP-DIR> --log_root <LOG-DIR> --tag cyolo`

By adapting this command it might be possible to also train across multiple nodes, but we only tested it for a single node.
Please check https://pytorch.org/tutorials/beginner/dist_overview.html and
https://pytorch.org/docs/stable/distributed.html for more information on distributed training.

## Evaluation
To reproduce the results shown in Table 2, we provide you with our trained models in the folder
[`trained_models`](https://github.com/CPJKU/cyolo_score_following/tree/eusipco-2021/trained_models).
To evaluate a single model on the test set you need to run the following command:

`python eval.py --param_path ../trained_models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/msmd_test --eval_onsets`

e.g., if you want to evaluate the conditional YOLO model trained with impulse response augmentation (CYOLO-IR), you need to execute:

`python eval.py --param_path ../trained_models/cyolo_ir/best_model.pt --test_dir ../data/msmd/msmd_test --eval_onsets`

If you want to print statistics for each piece and page separately you can add the `--print_piecewise` flag.
You can also evaluate only a subset from a provided directory by specifying an additional split file `--split_file ../data/msmd/split_files/<split>.yaml`.

## Visualization

To see what our network actually does, we can create a video of its performance on a certain piece:

``` python test.py --param_path ../trained_models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/<TEST-DIR> --test_piece <PIECE>```

There is also an optional `--page` parameter to specify which page of the piece should be evaluated,
e.g.,  if you want to create a video for the first page of the test piece *Anonymous__lanative__lanative* using our best performing model,
 you need to execute:
 
`python test.py --param_path ../trained_models/cyolo_ir/best_model.pt --test_dir ../data/msmd/msmd_test --test_piece Anonymous__lanative__lanative_synth --page 0`
 
 Note that if `--page` is not specified, all pages will be evaluated.

 ## Acknowledgements
This project has received funding from the European Research Council (ERC) 
under the European Union's Horizon 2020 research and innovation program
(grant agreement number 670035, project "Con Espressione"). 

<img src="https://erc.europa.eu/sites/default/files/LOGO_ERC-FLAG_EU_.jpg" width="35%" height="35%">
