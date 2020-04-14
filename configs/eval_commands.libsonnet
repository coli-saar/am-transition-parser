local ALTO_PATH = "am-tools.jar";
local WORDNET = "downloaded_models/wordnet3.0/dict/";

local tool_dir = "evaluation_tools/";

{
 "AMR-2015" : {
    "callbacks" : {
    "after_validation" : {
                 "type" : "parse-dev",
                 "system_input" : "data/AMR/2015/dev/dev.amconll",
                 "eval_command" : {
                     "type" : "bash_evaluation_command",
                     "gold_file" : "data/AMR/2015/dev/goldAMR.txt",
                     "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                         ' --lookup data/AMR/2015/lookup/ --th 10' +
                     '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                     "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                         "R" : [1, "Recall: (?P<value>.+)"],
                                         "F" : [2, "F-score: (?P<value>.+)"]}
             }
  }}
}

}