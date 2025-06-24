import re

from database_handler import DatabaseHandler
from deep_eval import ExecutionAccuracy, LLMTestCase


class SQLEvaluation(object):
    def __init__(self):
        self.correct_count = 0

    def get_sql_command_results(self, dataframe):

        # list_results = []
        self.total = dataframe.shape[0]

        for _, row in dataframe.iterrows():

            row["model_output"] = re.sub("<\|(.*)\|>", "", row['model_output'])

            if row["query"] == row["model_output"]:
                self.correct_count += 1
        
        return self.correct_count, self.total


    def get_sql_query_result(self, dataframe):
        
        list_results = []
        execution_metrics = ExecutionAccuracy()

        count = 0

        for _, row in dataframe.iterrows():

            _path = "/".join(["spider_data/database", row["db_id"], row["db_id"]+".sqlite"])
            local_database = DatabaseHandler(_path)
            model_query = re.sub("<\|(.*)\|>", "", row['model_output'])
            response = local_database.execute(model_query)

            question = row["question"]
            expected_output = row['response']
            actual_ouput = str(response)

            test_case = LLMTestCase(
                input=question,
                actual_output=actual_ouput,
                expected_output=expected_output,
            )

            execution_metrics.measure(test_case)

            if execution_metrics.is_successful():
                count += 1
            
        return count