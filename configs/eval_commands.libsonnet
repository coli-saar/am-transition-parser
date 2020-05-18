local ALTO_PATH = "am-tools.jar";
local WORDNET = "downloaded_models/wordnet3.0/dict/";

local tool_dir = "evaluation_tools/";
local data_paths = import 'data_paths.libsonnet';
local SDP_prefix = data_paths["SDP_prefix"];

local parse_test = true;

local SDP_regex = {
      "P" : [1, "Precision (?P<value>.+)"],
      "R" : [2, "Recall (?P<value>.+)"],
      "F" : [3, "F (?P<value>.+)"] #says: on line 3 (0-based), fetch the F-Score with the given regex.}
} ;

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
                  "result_regexes" : SDP_regex
         }
       },

       "after_training" : {
           "type" : "parse-test",
           "system_inputs" : [SDP_prefix+name+"/test.id/test.id.amconll", SDP_prefix+name+"/test.ood/test.ood.amconll"],
            "names" : [name+"_id", name+"_ood"],
            "active" : parse_test,
            "test_commands" : [
                {
                "type" : "bash_evaluation_command",
                "gold_file": SDP_prefix+name+"/test.id/en.id."+std.asciiLower(name)+".sdp",
                 "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                 "result_regexes" : SDP_regex
                 },
                 {
                 "type" : "bash_evaluation_command",
                 "gold_file": SDP_prefix+name+"/test.ood/en.ood."+std.asciiLower(name)+".sdp",
                  "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.dm.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                  "result_regexes" : SDP_regex
                 }
        ]
      }
   }

};

{
 "AMR-2015" : {
    "callbacks" : {
    "after_validation" : {
                 "type" : "parse-dev",
                 "system_input" : "data/AMR/2015/dev/dev.amconll",
                 "prefix": "AMR-2015_",
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
  },
   "after_training" : {
        "type" : "parse-test",
        "system_inputs" : ["data/AMR/2015/test/test.amconll"],
        "names" : ["AMR-2015"],
        "active" : parse_test,
        "test_commands" : [
            {
             "type" : "bash_evaluation_command",
             "gold_file" : "data/AMR/2015/test/goldAMR.txt",
              "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                  ' --lookup data/AMR/2015/lookup/ --th 10' +
              '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
              "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                  "R" : [1, "Recall: (?P<value>.+)"],
                                  "F" : [2, "F-score: (?P<value>.+)"]}
            }
        ]
   }

  }
},

 "AMR-2017" : {
    "callbacks" : {
    "after_validation" : {
                 "type" : "parse-dev",
                 "system_input" : "data/AMR/2017/dev/dev.amconll",
                 "prefix": "AMR-2017_",
                 "eval_command" : {
                     "type" : "bash_evaluation_command",
                     "gold_file" : "data/AMR/2017/dev/goldAMR.txt",
                     "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                         ' --lookup data/AMR/2017/lookup/ --th 10' +
                     '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                     "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                         "R" : [1, "Recall: (?P<value>.+)"],
                                         "F" : [2, "F-score: (?P<value>.+)"]}
             }
  },
     "after_training" : {
          "type" : "parse-test",
          "system_inputs" : ["data/AMR/2017/test/test.amconll"],
          "names" : ["AMR-2017"],
          "active" : parse_test,
          "test_commands" : [
              {
               "type" : "bash_evaluation_command",
               "gold_file" : "data/AMR/2017/test/goldAMR.txt",
                "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.amr.tools.EvaluateCorpus --corpus {system_output} -o {tmp}/ --relabel --wn '+WORDNET+
                    ' --lookup data/AMR/2017/lookup/ --th 10' +
                '&& python2 '+tool_dir+'/smatch/smatch.py -f {tmp}/parserOut.txt {gold_file} --pr --significant 4 > {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                "result_regexes" : {"P" : [0, "Precision: (?P<value>.+)"],
                                    "R" : [1, "Recall: (?P<value>.+)"],
                                    "F" : [2, "F-score: (?P<value>.+)"]}
              }
          ]
     }
  }
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
"PSD" :    { "callbacks" : {
                "after_validation" : {
                    "type" : "parse-dev",
                    "system_input" : SDP_prefix+"PSD/dev/dev.amconll",
                    "prefix": "PSD_",
                     "eval_command" : {
                         "type" : "bash_evaluation_command",
                         "gold_file": SDP_prefix+"PSD/dev/dev.sdp",
                          "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                          "result_regexes" : SDP_regex
                 }
               },
              "after_training" : {
                  "type" : "parse-test",
                  "system_inputs" : [SDP_prefix+"PSD/test.id/test.id.amconll", SDP_prefix+"PSD/test.ood/test.ood.amconll"],
                   "names" : ["PSD_id", "PSD_ood"],
                   "active" : parse_test,
                   "test_commands" : [
                       {
                       "type" : "bash_evaluation_command",
                       "gold_file": SDP_prefix+"PSD/test.id/en.id.psd.sdp",
                        "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                        "result_regexes" : SDP_regex
                        },
                        {
                        "type" : "bash_evaluation_command",
                        "gold_file": SDP_prefix+"PSD/test.ood/en.ood.psd.sdp",
                         "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.sdp.psd.tools.ToSDPCorpus --corpus {system_output} --gold {gold_file} --outFile {tmp}/BLABLA',
                         "result_regexes" : SDP_regex
                        }
               ]
             }
 } },

    "EDS" : { #don't use file extension for gold_file: use e.g. data/EDS/dev/dev-gold
        "callbacks" : {
        "after_validation" : {
                     "type" : "parse-dev",
                     "system_input" : "data/EDS/dev/dev.amconll",
                     "eval_command" : {
                        "type" : "bash_evaluation_command",
                        "gold_file": "data/EDS/dev/dev-gold",
                        "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus --corpus {system_output} --outFile {tmp}/output.eds'+
                        '&& python2 '+tool_dir+'/fast_smatch/fast_smatch.py -f {tmp}/output.eds.amr.txt {gold_file}.amr.txt --pr > {tmp}/metrics.txt'+
                        '&& python2 '+tool_dir+'/edm/eval_edm.py {tmp}/output.eds.edm {gold_file}.edm >> {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                        "result_regexes" : {"Smatch_P" : [0, "Precision: (?P<value>.+)"],
                                            "Smatch_R" : [1, "Recall: (?P<value>.+)"],
                                            "Smatch_F" : [2, "F-score: (?P<value>.+)"],
                                            "EDM_F" : [4,"F1-score: (?P<value>.+)"]}
                     }
         },
      "after_training" : {
               "type" : "parse-test",
               "system_inputs" : ["data/EDS/test/test.amconll"],
               "names" : ["EDS"],
               "active" : parse_test,
               "test_commands" : [
                  {
                        "type" : "bash_evaluation_command",
                        "gold_file": "data/EDS/test/test-gold",
                        "command" : 'java -cp '+ALTO_PATH+' de.saar.coli.amrtagging.formalisms.eds.tools.EvaluateCorpus --corpus {system_output} --outFile {tmp}/output.eds'+
                        '&& python2 '+tool_dir+'/fast_smatch/fast_smatch.py -f {tmp}/output.eds.amr.txt {gold_file}.amr.txt --pr > {tmp}/metrics.txt'+
                        '&& python2 '+tool_dir+'/edm/eval_edm.py {tmp}/output.eds.edm {gold_file}.edm >> {tmp}/metrics.txt && cat {tmp}/metrics.txt',
                        "result_regexes" : {"Smatch_P" : [0, "Precision: (?P<value>.+)"],
                                            "Smatch_R" : [1, "Recall: (?P<value>.+)"],
                                            "Smatch_F" : [2, "F-score: (?P<value>.+)"],
                                            "EDM_F" : [4,"F1-score: (?P<value>.+)"]}
                     }
               ]
          }
    }
    },

    "validation_metric" : {
        "AMR-2015" : "+AMR-2015_F",
        "AMR-2017" : "+AMR-2017_F",
        "DM" : "+DM_F",
        "PAS" : "+PAS_F",
        "PSD" : "+PSD_F",
        "EDS" : "+Smatch_F"
    }

}