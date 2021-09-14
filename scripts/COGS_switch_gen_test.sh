usage="NOTE: Switch to 'am-transition-parser/' to execute the script! \n
Switch between different gen and test. Overwrites test.amconll and test.tsv. \n\n
Arguments: \n
\n\t     -d  delete test.amconll and test.tsv
\n\t     -t  switch to gen/test. Options: gen, test.
"

while getopts "dg:h" opt; do
    case $opt in
        h)  echo -e $usage
	        exit
	        ;;
        d) rm data/COGS/test/current_test.amconll
           rm data/COGS/test/current_test.tsv
           exit
           ;;
	    g) testset="$OPTARG"
            ;;
        \?) echo "Invalid option -$OPTARG" >&2
	        ;;
    esac
done

cp data/COGS/test/$testset.amconll data/COGS/test/current_test.amconll
cp data/COGS/test/$testset.tsv data/COGS/test/current_test.tsv