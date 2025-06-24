import os
import pandas as pd

from metric_evaluation.metric_evaluation import MetricEvaluation
from pprint import pprint

from utils import list_dict_to_csv

def main():


    mmlu_context = ["astronomy", "high_school_biology", "prehistory"]

    list_final_results = []

    for context in mmlu_context:

        list_results = os.listdir("/".join(["results/eval", context]))
        mmlu_evaluation = MetricEvaluation()

        for _path in list_results:

            if "default_model" in _path:
               experiment = "baseline"
            elif "experiment-1" in _path:
                experiment = "experiment_1"
            elif "experiment-2" in _path:
                experiment = "experiment_2"
            elif "experiment-3" in _path:
                experiment = "experiment_3"
            elif "experiment-4" in _path:
                experiment = "experiment_4"
            else:
                experiment = "experiment_5"
            
            dataframe = pd.read_csv("/".join(["results/eval", context, _path]))
            
            mmlu_evaluation.execute(experiment, dataframe)
        
        dict_results = mmlu_evaluation.get_results()
        dict_results["type"] = context
        
        list_final_results.append(dict_results)
    
    list_dict_to_csv(list_final_results, "results/eval/mmlu_final_results.csv")


        


if __name__ == "__main__":
    main()