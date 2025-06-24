import pandas as pd
from metric_evaluation.sql_evaluation import SQLEvaluation

from utils import list_dict_to_csv


def main(list_results):

    final_results = []
    metrics_results = []

    for _results in list_results:
        sql_eval = SQLEvaluation()

        dataframe = pd.read_csv(_results["results"])
        correct, total = sql_eval.get_sql_command_results(dataframe)

        final_results.append(
            {
                "experiment": _results["experiment"],
                "correct": correct,
                "total": total,
                "accuracy": correct / total
            }            
        )
    
    for _results in list_results:
        sql_eval = SQLEvaluation()

        dataframe = pd.read_csv(_results["results"])
        total = dataframe.shape[0]
        output = sql_eval.get_sql_query_result(dataframe)

        metrics_results.append(
                {
                    "experiment": _results["experiment"],
                    "accuracy": output / total
                }
            )

    list_dict_to_csv(metrics_results, "results/query/final_results.csv")
    list_dict_to_csv(final_results, "results/sql_command_accuracy.csv")


if __name__ == "__main__":

    list_results = [
            {
                "experiment": "baseline",
                "results": "results/few_shot_results.csv"
            },
            {
                "experiment": "experiment_1", 
                "results": "results/fine_tuning_1_results--checkpoint--1014.csv"
            },
            {
                "experiment": "experiment_2",
                "results": "results/fine_tuning_2_results--checkpoint--2028.csv"
            },
            {
                "experiment": "experiment_3",
                "results": "results/fine_tuning_3_results--checkpoint--1521.csv"
            },
            {
                "experiment": "experiment_4",
                "results": "results/fine_tuning_4_results--checkpoint--1014.csv"
            },
            {
                "experiment": "experiment_5",
                "results":"results/fine_tuning_5_results--checkpoint--1183.csv"
            },
    ]

    main(list_results)