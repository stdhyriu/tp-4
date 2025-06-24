from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class ExecutionAccuracy(BaseMetric):

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):

        sql_model_output = test_case.actual_output
        sql_ground_truth = test_case.expected_output

        if sql_model_output == sql_ground_truth:
            self.score = 1.0
        else:
            self.score = 0.0

    def is_successful(self):

        return self.score >= self.threshold


if __name__ == "__main__":
    
    import pandas as pd

    dataframe = pd.read_csv("results/query/baseline.csv")
    execution_metrics = ExecutionAccuracy()

    for _, row in dataframe.iterrows():
        question = row["question"]
        expected_output = row['answer']
        actual_ouput = row['model_answer']

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_ouput,
            expected_output=expected_output,
        )

        execution_metrics.measure(test_case)

        print('resp', execution_metrics.is_successful())
        break