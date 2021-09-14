usage="NOTE: Switch to 'am-transition-parser/' to execute the script! \n
Switch between different gen and test. Overwrites test.amconll and test.tsv. \n\n
Arguments: \n
\n\t     -d  delete test.amconll and test.tsv
\n\t     -g  switch to gen
\n\t     -t  switch to real_test
"

while getopts "dgth" opt; do
    case $opt in
        h)  echo -e $usage
	        exit
	        ;;
        d) rm data/COGS/test/test.amconll
           rm data/COGS/test/test.tsv
           exit
           ;;
	    g) testset="gen"
	        ;;
        t) testset="real_test"
	        ;;
        \?) echo "Invalid option -$OPTARG" >&2
	        ;;
    esac
done

cp data/COGS/test/$testset.amconll data/COGS/test/test.amconll
cp data/COGS/test/$testset.tsv data/COGS/test/test.tsv