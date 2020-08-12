# Call with data/AMR/2015/
pushd $1
mkdir -p lexicon
cat train/train.amconll gold-dev/gold-dev.amconll | grep -vE "^#" | cut -f11 | sort | uniq | grep . > lexicon/edges.txt
cat train/train.amconll gold-dev/gold-dev.amconll | grep -vE "^#" | cut -f8 | sort | uniq | grep . > lexicon/lex_labels.txt
popd

python topdown_parser/tools/prepare_constants.py "$1/lexicon/" --corpora "$1/train/train.amconll" "$1/gold-dev/gold-dev.amconll"



