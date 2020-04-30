
git submodule update

# wordnet
wordnet_check="downloaded_models/wordnet3.0/dict/data.noun" # just check one file, should all be there or none
if [ -f "$wordnet_check" ]; then
    echo "WordNet found in downloaded_models/wordnet3.0 (If there are problems with missing files in that folder, delete it and run this script again.)"
else
    wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz -O tmp/WordNet-3.0.tar.gz
    tar -xzvf tmp/WordNet-3.0.tar.gz -C tmp
    mkdir -p downloaded_models/wordnet3.0
    mv tmp/WordNet-3.0/dict downloaded_models/wordnet3.0/
fi
rm -rf tmp

jar="am-tools.jar"
if [ -f "$jar" ]; then
    echo "jar file found at $jar"
else
    echo "jar file not found at $jar, downloading it!"
    wget -O "$jar" http://www.coli.uni-saarland.de/projects/amparser/am-tools.jar
fi

