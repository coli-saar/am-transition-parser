local ALTO_PATH = "am-tools.jar";
local WORDNET = "downloaded_models/wordnet3.0/dict/";

local tool_dir = "evaluation_tools/";
local data_paths = import 'data_paths.libsonnet';
local SDP_prefix = data_paths["SDP_prefix"];

local sdp_evaluator(name) = {
    "callbacks" : {
        "after_validation" : {
            "type" : "parse-dev",
            "system_input" : SDP_prefix+name+"/dev/dev.amconll",
            "prefix": name+"_",
             "eval_command" : {
                 "type" : "bash_evaluation_command",
                 "gold_file": SDP_prefix+name+"/dev/dev.sdp",
                  "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                  "result_regexes" : {
                                      "P" : [1, "Precision (?P<value>.+)"],
                                      "R" : [2, "Recall (?P<value>.+)"],
                                      "F" : [3, "F (?P<value>.+)"] #says: on line 3 (0-based), fetch the F-Score with the given regex.
                  }
         }
       }
   }

};

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
},

 "AMR-2017" : {
    "callbacks" : {
    "after_validation" : {
                 "type" : "parse-dev",
                 "system_input" : "data/AMR/2017/dev/dev.amconll",
                 "eval_command" : {
                     "type" : "bash_evaluation_command",
                     "gold_file" : "data/AMR/2017/dev/goldAMR.txt",
                     "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                         ' --lookup data/AMR/2015/lookup/ --th 10' +
                     '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                     "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                         "R" : [1, "Recall: (?P<value>.+)"],
                                         "F" : [2, "F-score: (?P<value>.+)"]}
             }
  }}
},

"general_validation" : {
   "type" : "bash_evaluation_command",
   "command" : "python3 topdown_parser/evaluation/am_dep_las.py {gold_file} {system_output}",
   "result_regexes" : {
       "Constant_Acc" : [4, "Supertagging acc % (?P<value>[0-9.]+)"],
       "Lex_Acc" : [5, "Lexical label acc % (?P<value>[0-9.]+)"],
       "UAS" : [6, "UAS.* % (?P<value>[0-9.]+)"],
       "LAS" : [7, "LAS.* % (?P<value>[0-9.]+)"],
       "Content_recall" : [8, "Content recall % (?P<value>[0-9.]+)"]
   }
},

"DM" : sdp_evaluator("DM"),
"PAS" : sdp_evaluator("PAS"),
"PSD" : sdp_evaluator("PSD"),

}