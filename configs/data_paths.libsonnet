local SDP_prefix = "data/SemEval/2015/";

{
    "GLOVE_DIR" : "/local/mlinde/glove/",

    "SDP_prefix" : SDP_prefix,

    "train_data" : {
        "DM" : SDP_prefix+"DM/train/train.amconll",
        "PAS" : SDP_prefix+"PAS/train/train.amconll",
        "PSD" : SDP_prefix+"PSD/train/train.amconll",
        "AMR-2015" : "data/AMR/2015/train/train.amconll",
        "AMR-2017" : "data/AMR/2017/train/train.amconll",
        "AMR-2020" : "data/AMR/2020/train/train.amconll",
        "EDS" : "data/EDS/train/train.amconll",
        "COGS" : "data/COGS/train/train.amconll",
    },
    "gold_dev_data" : { #gold AM dependency trees for (a subset of) the dev data
        "DM" : SDP_prefix+"DM/gold-dev/gold-dev.amconll",
        "PAS" : SDP_prefix+"PAS/gold-dev/gold-dev.amconll",
        "PSD" : SDP_prefix+"PSD/gold-dev/gold-dev.amconll",
        "AMR-2015" : "data/AMR/2015/gold-dev/gold-dev.amconll",
        "AMR-2017" : "data/AMR/2017/gold-dev/gold-dev.amconll",
        "AMR-2020" : "data/AMR/2020/gold-dev/gold-dev.amconll",
        "EDS" : "data/EDS/gold-dev/gold-dev.amconll",
        "COGS" : "data/COGS/gold-dev/gold-dev.amconll",
    }
}

