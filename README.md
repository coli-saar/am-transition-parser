# topdown-parser
A top-down transition-based AM dependency parser with quadratic run time complexity and well-typedness guarantees.
Implemented are the LTF and LTL transition systems and unconstrained versions for normal dependency parsing.

##Steps for setting up
- Create a conda environment with python 3.8
- `pip install -r requirements.txt`
- Copy the corpora with AM dependency trees to `data/`. They should be organized in the way the decomposition scripts in [am-parser](https://github.com/coli-saar/am-parser) create them (see also the wiki!). See the bottom of the page, which directory structure we assume.
- `bash scripts/setup.sh`, which will download am-tools (large file!) and WordNet.
- `bash scripts/create_all_lexica.sh` will create lexica with graph constants, edge labels and types. 

##Training a model
Select an appropriate configuration file (e.g. `training_configs/bert/DM.jsonnet`) and run the following command:
```
python -m allenpipeline train <your config.jsonnet> -s models/<your model name> --include-package topdown_parser
```
You can use other command line arguments as well, see `python -m allenpipeline train --help`, in particular you can select the cuda device as follows:
`-o '{trainer : {cuda_device : 0  } }'`.

##Parsing
There are different ways to parse, depending on what you want.

 - You want to parse the testset of a graphbank. Make sure, the folders in `data/` follow what the model expects (see `config.json` in the model directory -- you can also modify this file but create a backup!).
  Then you can parse by calling `python topdown_parser/parse_testset.py <your model> --cuda-device <device> --batch_size <batch size> --beams <list of beam sizes>`
  where `<list of beam sizes>` is simply `1` for greedy search, or for example `1 3` if you want to do greedy search AND beam search with beam size 3.
  This command will evaluate the AM dependency trees to graphs and compute F-scores with the gold standard.
 - You want to annotate an existing amconll file (with or without AM dependency trees in it). Then you should use the `topdown_parser/beam_search.py` script. Use the `--help` option to get information about how to structure the command line arguments.

Currently, this is no direct way to parse a raw text file. 
The way to go, would be to create an amconll file that contains the sentences in question but no AM dependency trees.


## Directory structure
By default, we assume the following directory structure. If you want to parse only the test set, you only need the `lexicon` subfolder, the `lookup
 ` subfolder (only for AMR) and the `test*` subfolders.
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
The lexica and some pre-trained models can be found in `/proj/irtg.shadow/EMNLP20/transition_systems/`.
