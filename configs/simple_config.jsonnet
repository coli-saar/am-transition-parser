
local num_epochs = 10;
local device = 0;

local word_dim = 128;
local char_dim = 16; #TODO Ma use 100
local num_filters = 50;
local filters = [3];
local max_filter = 3; //KEEP IN SYNC WITH filters!

local pos_embedding = 32;

local encoder_dim = 256;


local dropout_in = 0.33;

local eval_commands = import "eval_commands.libsonnet";

local additional_lexicon = {
     "sublexica" : {
            "edge_labels" : "data/AMR/2015/lexicon/edges.txt",
            "constants" : "data/AMR/2015/lexicon/constants.txt",
            "term_types" : "data/AMR/2015/lexicon/types.txt",
            "lex_labels" : "data/AMR/2015/lexicon/lex_labels.txt"
     }
} ;

local transition_system = {
//    "type" : "dfs-children-first",
//    "children_order" : "IO",
//    "reverse_push_actions" : false
    "type" : "ltl",
    "children_order" : "IO",
    "pop_with_0" : true,
    "additional_lexicon" : additional_lexicon,
};

local dataset_reader = {
               "type": "amconll",
               "transition_system" : transition_system,
               "workers" : 1,
               "overwrite_formalism" : "amr",
               "run_oracle" : true,
               "fuzz" : true,
               "token_indexers" : {
                    "tokens" : {
                        "type": "single_id",
                         "lowercase_tokens": true
                               },
                    "token_characters" : {
                        "type" : "characters",
                        "min_padding_length" : max_filter
                    }
               }


           };

local data_iterator = {
        "type": "same_formalism",
        "batch_size": 16,
       "formalisms" : ["amr"]
    };


{
    "dataset_reader": dataset_reader,
    "validation_dataset_reader" : dataset_reader,

    "validation_command" : {

        "type" : "bash_evaluation_command",
        "command" : "python3 topdown_parser/evaluation/am_dep_las.py {gold_file} {system_output}",

        "result_regexes" : {
            "Constant_Acc" : [4, "Supertagging acc % (?P<value>[0-9.]+)"],
            "Lex_Acc" : [5, "Lexical label acc % (?P<value>[0-9.]+)"],
            "UAS" : [6, "UAS.* % (?P<value>[0-9.]+)"],
            "LAS" : [7, "LAS.* % (?P<value>[0-9.]+)"],
            "Content_recall" : [8, "Content recall % (?P<value>[0-9.]+)"]
        }
    },



    "iterator": data_iterator,
    "model": {
        "type": "topdown",
        "transition_system" : transition_system,

        "input_dropout" : dropout_in,
        "encoder_output_dropout" : 0.2,

        "context_provider" : {
            "type" : "sum",
            "providers" : [
//                  {"type" : "type-embedder", "hidden_dim" : 2*encoder_dim, "additional_lexicon" : additional_lexicon },
//                                { "type" : "last-label-embedder",
//                                "additional_lexicon" : additional_lexicon,
//                                "hidden_dim" : 2*encoder_dim
//                                "dropout" : 0.2
//                                },
                  {"type" : "most-recent-child" }
            ]
        },

//        "tagger_context_provider" :{ "type" : "label-embedder",
//                                "additional_lexicon" : additional_lexicon,
//                                "hidden_dim" : 2*encoder_dim,
//                                "dropout" : 0.2
//        },

        "supertagger" : {
            "type" : "combined-tagger",
//            "type" : "no-decoder-tagger",
            "lexicon" : additional_lexicon,
            "namespace" : "constants",
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
//                "input_dim" : 2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : 1024,
                "dropout" : 0.0,
                "activations" : "tanh",
            }
        },

        "lex_label_tagger" : {
            "type" : "combined-tagger",
//            "type" : "no-decoder-tagger",
            "lexicon" : additional_lexicon,
            "namespace" : "lex_labels",
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
//                "input_dim" : 2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : 1024,
                "dropout" : 0.0,
                "activations" : "tanh",
            }
        },

//        "term_type_tagger" : {
//            "type" : "combined-tagger",
//            "lexicon" : additional_lexicon,
//            "namespace" : "term_types",
//            "mlp" : {
//                "input_dim" : 2*2*encoder_dim,
//                "num_layers" : 1,
//                "hidden_dims" : 1024,
//                "dropout" : 0.0,
//                "activations" : "tanh",
//            }
//        },

        "encoder" : {
            "type" : "lstm",
            "input_size" :  num_filters + word_dim + pos_embedding,
            "hidden_size" : encoder_dim,
            "bidirectional" : true,
        },


        "tagger_encoder" : {
            "type" : "lstm",
            "input_size" :  num_filters + word_dim + pos_embedding,
            "hidden_size" : encoder_dim,
            "bidirectional" : true,
        },

//        "tagger_decoder" : {
//            "type" : "identity",
//            "input_dim": 2*encoder_dim,
//            "hidden_dim" : 2*encoder_dim,
//        },

        "decoder" : {
            "type" : "ma-lstm",
            "input_dim": 2*encoder_dim,
            "hidden_dim" : 2*encoder_dim,
            "input_dropout" : 0.2,
            "recurrent_dropout" : 0.1
        },
        "text_field_embedder": {
               "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_dim
                },
            "token_characters": {
              "type": "character_encoding",
                  "embedding": {
                    "embedding_dim": char_dim
                  },
                  "encoder": {
                    "type": "cnn",
                    "embedding_dim": char_dim,
                    "num_filters": num_filters,
                    "ngram_filter_sizes": filters
                  },
              "dropout": dropout_in
            }
        },
        "edge_model" : {
//            "type" : "attention",
//            "attention" : {
//                "type" : "bilinear",
//                "vector_dim" : 2*encoder_dim,
//                "matrix_dim" : 2*encoder_dim
//            }
//            "type" : "ma",
//            "mlp" : {
//                    "input_dim" : 2*encoder_dim,
//                    "num_layers" : 1,
//                    "hidden_dims" : 256,
//                    "activations" : "elu",
//                    "dropout" : 0.1
//            }
            "type" : "mlp",
            "encoder_dim" : 2*encoder_dim,
            "hidden_dim" : 256,
//            "activation" : "elu"
        },
        "edge_label_model" : {
            #"type" : "ma",
            "type" : "simple",
            "lexicon" : additional_lexicon,
            "mlp" : {
                #"input_dim" : 2*encoder_dim,
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : [256],
                "activations" : "tanh",
                "dropout" : 0.2
            }
        },
        "edge_loss" : {
            "type" : "nll"
        },

        "pos_tag_embedding" : {
            "embedding_dim" : pos_embedding,
            "vocab_namespace" : "pos"
        }

    },
    "train_data_path": "data/AMR/2015/train/small.amconll",
    "validation_data_path": "data/AMR/2015/gold-dev/gold-dev.amconll",

    "evaluate_on_test" : false,

    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
            "betas" : [0.9, 0.9]
        },
        "num_serialized_models_to_keep" : 1,
        "epochs_before_validate" : 2,
        "validation_metric" : "+LAS"
    },

    "dataset_writer":{
      "type":"amconll_writer"
    },

    "annotator" : {
        "dataset_reader": dataset_reader,
        "data_iterator": data_iterator,
        "dataset_writer":{
              "type":"amconll_writer"
        }
    },

//    "callbacks" : eval_commands["AMR-2015"]
}

