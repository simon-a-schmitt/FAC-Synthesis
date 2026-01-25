import time
import re
import string
import json
import concurrent
import multiprocessing
import tqdm
from OpenaiAPI import Chatting


class HelpfulJudge:
    def __init__(self, key):
        instruct = "You are an expert in evaluating Large Language Models. " +\
                   "__Your job is to identify whether specific text-based features indicate a strong capability for Instruction Following, according to the provided Guideline.__ " +\
                   "You are given a feature with its name and several text spans from a user-chatbot conversation (mostly chatbot outputs) as examples. " +\
                   "Analyze whether the presence of this feature suggests the chatbot is correctly following user instructions, constraints, or formatting requirements. " +\
                   "Note that, the duplicated text spans in the example are acceptable. " +\
                   "Please do not be too mean but be as subjective as possible based on the criteria. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Maybe/No ]]``. " +\
                   "\n\n__Guideline of Instruction Following (AlpacaEval 2.0 Criteria)__\n" +\
                   "* The feature does not need to satisfy all the following rules, but should align with the general goal of precise execution.\n" +\
                   "* The feature shows the chatbot adhering to specific constraints (e.g., word count limits, specific start/end phrases, negative constraints like 'do not mention').\n" +\
                   "* The feature demonstrates structured formatting requested by prompts (e.g., generating valid JSON, Markdown tables, bullet points, or code blocks).\n" +\
                   "* The feature indicates the chatbot is addressing ALL parts of a complex, multi-step user query, not just the first part.\n" +\
                   "* The feature shows the chatbot adopting a specific persona or style as requested (e.g., 'speak like a pirate', 'be professional').\n" +\
                   "* The feature reflects conciseness and directness. (Note: AlpacaEval 2.0 favors direct answers over overly long, rambling, or sycophantic responses).\n" +\
                   "* The feature involves reasoning or logic required to execute the instruction correctly (e.g., step-by-step thinking to solve a math puzzle)."

        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                        temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").replace("<s>[INST]", "").strip()
        case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
        if 'cannot tell' in case[0].lower():
            return "Example Text Spans: \n%s" % case[1]
        return "Feature Name: %s\nExample Text Spans: \n%s" % tuple(case)

    def clean(self, verify):
        temp = verify
        verify = verify[0].lower().split("decision")[-1]
        if "[[" in verify and "]]" in verify:
            verify = verify.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
        if verify.startswith(": "):
            verify = verify[2:].split(".", 1)[0].strip()
        if 'no' not in verify:
            print(temp)
        return verify + "|||" + temp[0]
    

if __name__ == "__main__":
    KEY = os.environ.get("OPENAI_API_KEY")
    
    model = HelpfulJudge(KEY)

    import sys
    file = sys.argv[1]
    print("Judging File: %s" % file)
    results = []
    with open(file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        assert headline == ["FeatureID", "Verify", "Summary", "Words"]
        for row in f.read().strip('\n').split("\n"):
            row = [_.replace("\\n", '\n').replace('\\t', '\t') for _ in row.split("\t")]
            results.append(['-'] + row)
    
    need_judge = [item for item in results if 'span' in item[4].lower() and 'cannot tell' not in item[3]]
    relations = model([_[3], _[4]] for _ in need_judge)
    for item, rela in zip(need_judge, relations):
        item[0] = rela
        

    from collections import Counter
    c = Counter([_[0].split("|||")[0] for _ in results])
    print("Explainability: %.4f" % ((c["yes"] + c["probably"]) / len(results)))
    for cate, freq in c.items():
        print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_AlpacaEval2.0.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tTask\tVerify\tSummary\tWords\n")
        for task, idx, verify, summary, words in results:
            task = task.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\t%s\n" % (idx, task, verify, summary, words))

