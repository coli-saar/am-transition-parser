local num_epochs = 100;
local device = 2;

local word_dim = 200;
local pos_embedding = 32;

local encoder_dim = 512;
local batch_size = 64;

local char_dim = 100;
local num_filters = 50;
local filters = [3];
local max_filter = 3; //KEEP IN SYNC WITH filters!

local bert_model = "bert-large-uncased";
local transformer_output_dim = 1024;

local lemma_embedding = 64;

local ne_embedding = 16;

local lexicon = import "../../configs/lexicon.libsonnet";

local eval_commands = import "../../configs/eval_commands.libsonnet";

local data_paths = import "../../configs/data_paths.libsonnet";

local task = "AMR-2017";

local transition_system = {
    "type" : "ltl",
    "children_order" : "IO",
    "pop_with_0" : true,
    "additional_lexicon" : lexicon[task],
};

local dataset_reader = {
               "type": "amconll",
               "transition_system" : transition_system,
               "workers" : 8,
               "run_oracle" : true,
               "overwrite_formalism" : task,

              "token_indexers" : {
                    "bert": {
                      "type": "pretrained_transformer_mismatched",
                      "model_name": bert_model,
                      "namespace" : "bert_tags"
                    },
                   "token_characters" : {
                       "type" : "characters",
                       "min_padding_length" : max_filter
                   }
              }

           };

local data_iterator = {
        "batch_sampler" : {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys" : ["words"]
        }
    };


{
    "dataset_reader": dataset_reader,
    "validation_dataset_reader" : dataset_reader,

     "vocabulary" : {
            "min_count" : {
            "lemmas" : 7
           }
     },

    "validation_command" : eval_commands["general_validation"],

    "data_loader": data_iterator,
    "model": {
        "type": "topdown",
        "transition_system" : transition_system,

        "context_provider" : {
            "type" : "plain-concat",
            "providers" : [
                   {"type" : "most-recent-child"},
                   {"type" : "parent" }
                   ]
        },

        "input_dropout" : 0.33,
        "encoder_output_dropout" : 0.33,

            "supertagger" : {
//                "type" : "simple-tagger",
                "type" : "combined-tagger",
                "lexicon" : lexicon[task],
                "namespace" : "constants",
                "mlp" : {
                    "input_dim" : 2*2*encoder_dim,
                    "num_layers" : 1,
                    "hidden_dims" : 1024,
                    "dropout" : 0.4,
                    "activations" : "tanh",
                }
            },

            "lex_label_tagger" : {
            "type" : "combined-tagger",
//            "type" : "no-decoder-tagger",
            "lexicon" : lexicon[task],
            "namespace" : "lex_labels",
                "mlp" : {
                    "input_dim" : 2*2*encoder_dim,
                    "num_layers" : 1,
                    "hidden_dims" : 1024,
                    "dropout" : 0.4,
                    "activations" : "tanh",
                }
            },

        "encoder" : {
             "type": "stacked_bidirectional_lstm",
            "input_size": num_filters + transformer_output_dim + pos_embedding + ne_embedding + lemma_embedding,
            "hidden_size": encoder_dim,
            "num_layers" : 3,
            "recurrent_dropout_probability" : 0.33,
            "layer_dropout_probability" : 0.33
        },

        "tagger_encoder" : {
             "type": "stacked_bidirectional_lstm",
            "input_size": num_filters + transformer_output_dim + pos_embedding + ne_embedding + lemma_embedding,
            "hidden_size": encoder_dim,
            "num_layers" : 1,
            "recurrent_dropout_probability" : 0.33,
            "layer_dropout_probability" : 0.33
        },

        "decoder" : {
            "type" : "ma-lstm",
            "input_dim": 3*2*encoder_dim,
            "hidden_dim" : 2*encoder_dim,
            "input_dropout" : 0.33,
            "recurrent_dropout" : 0.33
        },
           "text_field_embedder": {
               "type": "basic",
               "token_embedders": {
                   "bert" : {
                       "type": "scalar_mix_pretrained_transformer_mismatched",
                        "model_name" : bert_model,
                        "train_parameters" : false,
                       },
                "token_characters": {
                     "type": "character_encoding",
                         "embedding": {
                           "embedding_dim": char_dim,
                           "vocab_namespace": "token_characters",
                         },
                         "encoder": {
                           "type": "cnn",
                           "embedding_dim": char_dim,
                           "num_filters": num_filters,
                           "ngram_filter_sizes": filters
                         }
                   }
                }
           },

        "edge_model" : {
            "type" : "ma",
            "mlp" : {
                    "input_dim" : 2*encoder_dim,
                    "num_layers" : 1,
                    "hidden_dims" : 512,
                    "activations" : "elu",
                    "dropout" : 0.33
            }
        },

        "edge_label_model" : {
            "type" : "simple",
            "lexicon" : lexicon[task],
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : [256],
                "activations" : "tanh",
                "dropout" : 0.33
            }
        },

        "edge_loss" : {
            "type" : "nll"
        },

        "pos_tag_embedding" : {
            "embedding_dim" : pos_embedding,
            "vocab_namespace" : "pos"
        },

        "ne_embedding" : {
            "embedding_dim" : ne_embedding,
            "vocab_namespace" : "ner"
        },

        "lemma_embedding" : {
            "embedding_dim" : lemma_embedding,
            "vocab_namespace" : "lemmas"
        }

    },
    "train_data_path": data_paths["train_data"][task],
    "validation_data_path": data_paths["gold_dev_data"][task],

    "evaluate_on_test" : false,

       "trainer": {
           "type": "pipeline",
           "num_epochs": num_epochs,
           "cuda_device": device,
           "optimizer": {
               "type": "adam",
               "betas" : [0.9, 0.9]
           },
           "checkpointer" : {
           "num_serialized_models_to_keep" : 1,
           },
           "validation_metric" : eval_commands["validation_metric"][task],

           "external_callbacks" : eval_commands[task]
       },

       "dataset_writer":{
         "type":"amconll_writer"
       },

       "annotator" : {
           "type": "default",
           "dataset_reader": dataset_reader,
           "data_loader": data_iterator,
           "dataset_writer":{
                 "type":"amconll_writer"
           }
       }

}
