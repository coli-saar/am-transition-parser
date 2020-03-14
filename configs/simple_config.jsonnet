
local num_epochs = 50;
local device = 0;

local word_dim = 128;
local pos_embedding = 32;

local encoder_dim = 256;


local dataset_reader = {
               "type": "amconll",
               "transition_system" : {
                "type" : "dfs"
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
        "command" : "python topdown_parser/evaluation/am_dep_las.py {gold_file} {system_output}",

        "result_regexes" : {
            "UAS" : [6, "UAS.* % (?P<value>[0-9.]+)"],
            "LAS" : [7, "LAS.* % (?P<value>[0-9.]+)"]
        }
    },



    "iterator": data_iterator,
    "model": {
        "type": "topdown",
        "transition_system" : {
            "type" : "dfs"
        },

//        "context_provider" : {
//            "type" : "parents"
//        },

        "encoder" : {
            "type" : "lstm",
            "input_size" : word_dim + pos_embedding,
            "hidden_size" : encoder_dim,
            "bidirectional" : true,
        },
        "decoder" : {
            "type" : "lstm_cell",
            "input_dim": 2*encoder_dim,
            "hidden_dim" : 2*encoder_dim
        },
        "text_field_embedder": {
               "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_dim
                },
        },
        "edge_model" : {
            "type" : "attention",
            "attention" : {
                "type" : "bilinear",
                "vector_dim" : 2*encoder_dim,
                "matrix_dim" : 2*encoder_dim
            }
        },
        "edge_label_model" : {
            "type" : "simple",
            "formalism" : "amr",
            "mlp" : {
                "input_dim" : 2*2*encoder_dim,
                "num_layers" : 1,
                "hidden_dims" : [128],
                "activations" : "tanh"
            }
        },

        "pos_tag_embedding" : {
            "embedding_dim" : pos_embedding,
            "vocab_namespace" : "pos"
        }

    },
    "train_data_path": "data/tratz/gold-dev/gold-dev.amconll",
    "validation_data_path": "data/tratz/gold-dev/gold-dev.amconll",

    "evaluate_on_test" : false,

    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "num_serialized_models_to_keep" : 1,
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
    }
}

