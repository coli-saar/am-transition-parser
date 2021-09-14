# am-transition-parser
A top-down transition-based AM dependency parser with quadratic run time complexity and well-typedness guarantees.
Implemented are the LTF and LTL transition systems and unconstrained versions for normal dependency parsing. (If you are working on **COGS** see notes below.)

## Steps for setting up
- Create a conda environment with python 3.8
- `pip install -r requirements.txt`
- Copy the corpora with AM dependency trees to `data/`. They should be organized in the way the decomposition scripts in [am-parser](https://github.com/coli-saar/am-parser) create them (see also the wiki!). See the bottom of the page, which directory structure we assume.
- `bash scripts/setup.sh`, which will download am-tools (large file!) and WordNet.
- `bash scripts/prepare_all_lexica.sh` will create lexica with graph constants, edge labels and types. 

## Training a model
Select an appropriate configuration file (e.g. `training_configs/bert/DM.jsonnet`) and run the following command:
```
python -m allenpipeline train <your config.jsonnet> -s models/<your model name> --include-package topdown_parser
```
You can use other command line arguments as well, see `python -m allenpipeline train --help`, in particular you can select the cuda device as follows:
`-o '{trainer : {cuda_device : 0  } }'`.

You can train an almost minimal configuration with the provided example AM dependency trees as follows:
```
python -m allenpipeline train configs/example_config.jsonnet -s models/example-model --include-package topdown_parser
```

## Parsing
There are different ways to parse, depending on what you want.

 - You want to parse the testset of a graphbank. Make sure, the folders in `data/` follow what the model expects (see `config.json` in the model directory -- you can also modify this file but create a backup!).
  Then you can parse by calling `python topdown_parser/parse_testset.py <your model> --cuda-device <device> --batch_size <batch size> --beams <list of beam sizes>`
  where `<list of beam sizes>` is simply `1` for greedy search, or for example `1 3` if you want to do greedy search AND beam search with beam size 3.
  This command will evaluate the AM dependency trees to graphs and compute F-scores with the gold standard.
 - You want to annotate an existing amconll file (with or without AM dependency trees in it). Then you should use the `topdown_parser/beam_search.py` script. Use the `--help` option to get information about how to structure the command line arguments.
- You want to parse a raw text file. You can create an amconll file without AM dependency trees in it using the `raw_to_amconll.py` script in [am-parser](https://github.com/coli-saar/am-parser). 
 **Beware**: this is not the way we prepared the test sets in our experiments, and you should consider using a raw-text model, that is a model which does not actually use the POS tags, lemmas and named entity tags in the amconll file (this is achieved by using a configuration where the embedding size is 0 for those embedding types).


## Directory structure
By default, we assume the following directory structure. If you want to parse only the test set, you only need the `lexicon` subfolder, the `lookup` subfolder (only for AMR) and the `test*` subfolders.
```
data/
├── AMR
│   ├── 2015
│   │   ├── dev
│   │   │   ├── dev.amconll
│   │   │   └── goldAMR.txt
│   │   ├── gold-dev
│   │   │   └── gold-dev.amconll
│   │   ├── lexicon
│   │   │   ├── constants.txt
│   │   │   ├── edges.txt
│   │   │   ├── lex_labels.txt
│   │   │   └── types.txt
│   │   ├── lookup
│   │   │   ├── nameLookup.txt
│   │   │   ├── nameTypeLookup.txt
│   │   │   ├── README.txt
│   │   │   ├── wikiLookup.txt
│   │   │   └── words2labelsLookup.txt
│   │   ├── test
│   │   │   ├── goldAMR.txt
│   │   │   └── test.amconll
│   │   └── train
│   │       └── train.amconll
│   └── 2017
│       ├── dev
│       │   ├── dev.amconll
│       │   └── goldAMR.txt
│       ├── gold-dev
│       │   └── gold-dev.amconll
│       ├── lexicon
│       │   ├── constants.txt
│       │   ├── edges.txt
│       │   ├── lex_labels.txt
│       │   └── types.txt
│       ├── lookup
│       │   ├── nameLookup.txt
│       │   ├── nameTypeLookup.txt
│       │   ├── wikiLookup.txt
│       │   └── words2labelsLookup.txt
│       ├── test
│       │   ├── goldAMR.txt
│       │   └── test.amconll
│       └── train
│           └── train.amconll
├── EDS
│   ├── dev
│   │   ├── dev.amconll
│   │   ├── dev-gold
│   │   ├── dev-gold.amr.txt
│   │   └── dev-gold.edm
│   ├── gold-dev
│   │   └── gold-dev.amconll
│   ├── lexicon
│   │   ├── constants.txt
│   │   ├── edges.txt
│   │   ├── lex_labels.txt
│   │   └── types.txt
│   ├── README.txt
│   ├── test
│   │   ├── test
│   │   ├── test.amconll
│   │   ├── test-gold
│   │   ├── test-gold.amr.txt
│   │   └── test-gold.edm
│   └── train
│       ├── train.amconll
└── SemEval
    └── 2015
        ├── DM
        │   ├── dev
        │   │   ├── dev.amconll
        │   │   └── dev.sdp
        │   ├── gold-dev
        │   │   └── gold-dev.amconll
        │   ├── lexicon
        │   │   ├── constants.txt
        │   │   ├── edges.txt
        │   │   ├── lex_labels.txt
        │   │   └── types.txt
        │   ├── test.id
        │   │   ├── en.id.dm.sdp
        │   │   └── test.id.amconll
        │   ├── test.ood
        │   │   ├── en.ood.dm.sdp
        │   │   └── test.ood.amconll
        │   └── train
        │       └── train.amconll
        ├── PAS
        │   ├── dev
        │   │   ├── dev.amconll
        │   │   └── dev.sdp
        │   ├── gold-dev
        │   │   └── gold-dev.amconll
        │   ├── lexicon
        │   │   ├── constants.txt
        │   │   ├── edges.txt
        │   │   ├── lex_labels.txt
        │   │   └── types.txt
        │   ├── test.id
        │   │   ├── en.id.pas.sdp
        │   │   └── test.id.amconll
        │   ├── test.ood
        │   │   ├── en.ood.pas.sdp
        │   │   └── test.ood.amconll
        │   └── train
        │       └── train.amconll
        └── PSD
            ├── dev
            │   ├── dev.amconll
            │   └── dev.sdp
            ├── gold-dev
            │   ├── gold-dev.amconll
            ├── lexicon
            │   ├── constants.txt
            │   ├── edges.txt
            │   ├── lex_labels.txt
            │   └── types.txt
            ├── test.id
            │   ├── en.id.psd.sdp
            │   └── test.id.amconll
            ├── test.ood
            │   ├── en.ood.psd.sdp
            │   └── test.ood.amconll
            └── train
                └── train.amconll


```

## Uni Saarland internal notes
The lexica and some pre-trained models can be found in `/proj/irtg.shadow/EMNLP20/transition_systems/`. A conda environment is prepared already and it's called `pytorch1.4`.

## Training the parser on COGS

1. __Setup:__
    - Create a conda environment with python 3.8 (or use the environment on the Saarland servers, see above)
    - `pip install -r requirements.txt`
    - get the COGS-aware version of am-tools.jar from https://github.com/coli-saar/am-tools/tree/cogs_new_decomp. __Note:__ If you are working on the Saarland servers you can copy it from `/proj/irtg/sempardata/cogs2021/jar/am-tools.jar`.
    - `bash scripts/setup.sh`, which will download WordNet (download of am-tools is skipped).

2. __Training data:__ 
    - We currently use the output of the unsupervised parser (best predictions of the model are written to a separate amconll file each epoch) as the input for the 'supervised' parsers. This means all amconll files have to be created using the unsupervised parser. 
    - The gold data in original COGS format (TSV files) can be found on https://github.com/najoungkim/COGS. 
    - __Note:__ If you are working on the Saarland servers the TSV files can be found in `/proj/irtg/sempardata/cogs2021/data/COGS/data`. Different versions of the amconll files might be in `/proj/irtg/sempardata/cogs2021/retrain/input`.

3. There are __different training sets__ (train and train_100) for COGS. To switch between the different training sets / embeddings / outputs of the unsupervised parser use the script `scripts/COGS_switch_train_set.sh -n <name of train set x>` This script creates the `train.amconll`, `dev.amconll`, and `gold-dev.amconll` needed to train the parser. It also creates a new lexicon. If such files already exist, they are overwritten. The script assumes the following structure:
    ```
    data/
    └── COGS
        ├── dev
        │   ├── dev_{name of train set 1}.amconll
        │   ├── dev_{name of train set 2}.amconll
        │   ├── ...
        │   ├── dev_{name of train set x}.amconll
        │   ├── dev.amconll                             //if this file exists, it will be overwritten
        │   └── dev.tsv                                 //only needed later for training
        ├── gold-dev
        │   ├── gold-dev_{name of train set 1}.amconll
        │   ├── gold-dev_{name of train set 2}.amconll
        │   ├── ...
        │   ├── gold-dev_{name of train set x}.amconll
        │   └── gold-dev.amconll                        //if this file exists, it will be overwritten
        ├── lexicon
        │   └── ...                                     //if these files exists, they will be overwritten
        ├── train
        │   ├── train_{name of train set 1}.amconll
        │   ├── train_{name of train set 2}.amconll
        │   ├── ...
        │   ├── train_{name of train set x}.amconll
        │   └── train.amconll                           //if this file exists, it will be overwritten
        └── test
            └── ...                                     //only needed later for testing
    ```
4. __Train__ the parser using the following command.
    ```
    python -m allenpipeline train training_configs/bert/COGS.jsonnet -s models/<your model name> --include-package topdown_parser
    ```
    Make sure you have the right train and dev files (see above). 

    __Notes:__ 
    - If `models/<your model name>` already exists you will get an error. 
    - On Saarland servers, training COGS on GPU currently does not work. Therefore you need to set `-o '{trainer : {cuda_device : -1 } }'`. 
    - If you abort training early, the script will still save a model to `models/<your model name>/model.tar.gz`. On COGS, it may be good to keep training running a little longer even if performance does not seem to improve anymore.

5. __Testing:__ There are two different test sets in the COGS data set: the real test set and the generalization set.      
    - Again, the gold data in original COGS format (TSV files) can be found on https://github.com/najoungkim/COGS. 
    - Additionally, you need empty amconll files. These can be created from the gold data using `raw_to_amconll.py` (see paragraph in [Parsing](https://github.com/coli-saar/am-transition-parser#parsing) above). The resulting files might contain errors. __Notes:__ If you are working on the Saarland servers, you can find the empty amconll files in `/proj/irtg/sempardata/cogs2021/retrain/`.
    - To run tests, save the tsv and amconll files in 
    ```
    data/COGS/test/
    ├── test.amconll
    ├── test.tsv
    ├── gen.amconll
    └── gen.tsv
    ```
    Then run `bash scripts/COGS_switch_gen_test.sh -t <gen or test>` to create the files `data/COGS/test/current_test.amconll` and `data/COGS/test/current_test.tsv` from gen or test. If these files already exist, they will be overwritten.
    - Test the model (also see paragraph [Parsing](https://github.com/coli-saar/am-transition-parser#parsing) above). 
    ```
    python topdown_parser/parse_testset.py <your model> --batch_size <batch size> --beams <list of beam sizes> [--cuda-device <device>]
    ```
    This will test `<your model>` on the `current_test` set and compute *ExactMatch* (the exact match accuracy on the logical forms) and *EditDistance*. __Notes:__ The performance on the real test set (in distribution) should be close to 100; the performance on the generalization set will be lower. --- Curiously, parsing on GPU is not a problem for Saarland servers.

6. __Misc:__
    - On COGS, we don't have PoS tag, Lemma, Named entity information available (columns empty in amconll, no embeddings for these, no extra files as input). 
    - Some of the training samples are 1-word sentences. To deal with these primitives, we commented out lines 184-236 in `topdown_parser/dataset_readers/amconll.py`. __Be aware of this if you want to train this branch of the parser on other formalisms / want to merge this branch in the future.__