class MetricEvaluation(object):
    
    def __init__(self):
        
        self.dict_results = {
            "baseline": 0,
            "experiment_1": 0,
            "experiment_2": 0,
            "experiment_3": 0,
            "experiment_4": 0,
            "experiment_5": 0,
        }

    def execute(self, experiment, dataframe):

        for _, row in dataframe.iterrows():
            
            if str(row["answer"]) == str(row["model_output"]).replace("### Answer:\n",""):
                self.dict_results[experiment] += 1
    
    def get_results(self,):
        return self.dict_results