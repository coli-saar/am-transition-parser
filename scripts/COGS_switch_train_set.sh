usage="Switch between different training sets for COGS. Overwrites current train.amconll, dev.amconll, and gold-dev.amconll.\n\n
Arguments: \n
\n\t     -n  name of the COGS training set. Options: trainBert, train100Bert, trainToken, train100Token
"

while getopts "n:h" opt; do
    case $opt in
        h)  echo -e $usage
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