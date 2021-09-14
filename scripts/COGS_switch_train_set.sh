usage="NOTE: Switch to 'am-transition-parser/' to execute the script! \n 
Switch between different training sets for COGS. Overwrites current train.amconll, dev.amconll, gold-dev.amconll, and the lexicon.\n\n
Arguments: \n
\n\t	 -d  delete train.amconll, dev.amconll, gold-dev.amconll, and the lexicon
\n\t     -n  name of the COGS training set. Options: trainBert, train100Bert, trainToken, train100Token \n
"

while getopts "dn:h" opt; do
    case $opt in
        h)  echo -e $usage
	        exit
	        ;;
		d) 	rm data/COGS/train/train.amconll
			rm data/COGS/dev/dev.amconll
			rm data/COGS/gold-dev/gold-dev.amconll
			rm data/COGS/lexicon/*
			exit
			;;
	    n) trainsetname="$OPTARG"
	        ;;
        \?) echo "Invalid option -$OPTARG" >&2
	    ;;
    esac
done

cp data/COGS/train/train_$trainsetname.amconll data/COGS/train/train.amconll
cp data/COGS/dev/dev_$trainsetname.amconll data/COGS/dev/dev.amconll
cp data/COGS/gold-dev/gold-dev_$trainsetname.amconll data/COGS/gold-dev/gold-dev.amconll
rm data/COGS/lexicon/*
bash scripts/prepare_all_lexica.sh