import pandas as pd
import numpy as np
from tabulate import tabulate
from valentine.algorithms import Coma
from valentine import valentine_match
import logging
import glob
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import warnings
from typing import Set
import random
import networkx as nx
import tempfile
from HelperFunctions import get_df_with_prefix, pearson_correlation
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

logging.basicConfig(level=logging.INFO)

class Join():

    def __init__(self, from_table: str, to_table: str, from_col: str, to_col: str, data_quality: float):
        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.data_quality = data_quality
        
    from_table: str
    to_table: str
    from_col: str
    to_col: str
    data_quality : float
    rel_red: dict

    def __str__(self) -> str:
        return "Join from " + self.from_table + "." + self.from_col + " to " + self.to_table +  "." + self.to_col + " with data quality " + str(self.data_quality) + " and rel_red " + str(self.rel_red)

class Path():

    def __init__(self, begin: str, path: [Join], rank: float):
        self.begin = begin
        self.path = path
        self.rank = rank

    def getBegin(self):
        return self.begin
    
    def getPath(self):
        return self.path
    
    def getRank(self):
        return self.rank
    
    def setBegin(self, begin: str):
        self.begin = begin

    def setPath(self, path: [Join]):
        self.path = path
    
    def setRank(self, rank: float):
        self.rank = rank
    
    def getRel_Red(self):
        dict = {}
        i: Join
        for i in self.path:
            for rel_red in i.rel_red:
                dict[i.to_table + "." + rel_red] = i.rel_red[rel_red]
                dict[i.to_table + "." + rel_red]["data_quality"] = i.data_quality
        return dict
    
    def __str__(self) -> str:
        ret_string = "Begin: " + self.begin
        for i in self.path:
            ret_string += "\n \t" + str(i) 
        ret_string += "\n \t rank: " + str(self.rank)
        return ret_string
    
    def explain(self) -> str:
        ret_str = "The path starts at the table " + self.begin + ". \n The path has the following joins: "
        for i in self.path:
            ret_str += "\n \t from (table.column) " + i[0] + " to  (table.column)" + i[1]
        ret_str += "\n The rank of the path is " + str(self.rank) + "."
        return ret_str
    
class Paths():
    paths: [Path]

    def __init__(self):
        self.paths = []

    def getPaths(self):
        return self.paths
    
    def addPath(self, path: Path):
        self.paths.append(path)
    
    def addPaths(self, paths: [Path]):
        self.paths.extend(paths)
    
    def __str__(self) -> str:
        ret_str = "Paths:"
        for i in self.paths:
            ret_str += "\n" + str(i)
        return ret_str
    
class Result():

    rank: float
    path: Path
    accuracy: float
    feature_importance: dict
    model: str
    data: pd.DataFrame

    def __init__(self):
        self.feature_importance = {}

    def getRank(self):
        return self.rank
    
    def getPath(self):
        return self.path
    
    def getAccuracy(self):
        return self.accuracy
    
    def setRank(self, rank: float):
        self.rank = rank
    
    def setPath(self, path: list):
        self.path = path
    
    def setAccuracy(self, accuracy: float):
        self.accuracy = accuracy

    def setFeatureImportance(self, feature_importance: dict):
        self.feature_importance = feature_importance
    
    def setModel(self, model: str):
        self.model = model
    
    def setData(self, data: pd.DataFrame):
        self.data = data
    

    def getFeatureImportance(self, tableName = None, featureName = None):
        if tableName is None and featureName is None:
            return self.result.feature_importance
        elif tableName is not None and featureName is None:
            return self.getFeatureImportanceByTable(self, tableName)
        elif tableName is not None and featureName is not None:
            return self.getFeatureImportanceByTableAndFeature(self, tableName, featureName)
    
    def getFeatureImportanceByTable(self, tableName: str):
        list = []
        for i in self.feature_importance:
            if tableName in i:
                list.append(i)
        return list
    
    def getFeatureImportanceByTableAndFeature(self, tableName: str, featureName: str):
        return self.feature_importance[tableName + "." + featureName]
    
    def showTable(self):
        scores = self.path.getRel_Red()
        table_data = []
        for key, values in scores.items():
            row = [key, values['rel'], values['red'], values["data_quality"]]
            table_data.append(row)
        # Displaying the table
        print(self.path)
        table = tabulate(table_data, headers=["Key", "Relevance", "Redundancy", "Data Quality"], tablefmt="grid")
        print(table)

    def showGraph(self):
        G = nx.Graph()
        for i in self.path.path:
            G.add_edge(i.from_table, i.to_table)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    def __str__(self) -> str:
        ret_string = "Result with model" + self.model + "\n \t rank:" + str(self.rank) + " \n \t with path:" + str(self.path) + " \n \t Accuracy:" + str(self.accuracy) + " \n \t Feature importance:"
        for i in self.feature_importance[0]:
            ret_string += "\n \t \t " + str(i) + " : " + str(self.feature_importance[0][i])
        return ret_string
    
    def why(self) -> str:
        return "The result is calculated by evaluating the path with the AutoML algorithm AutoGluon. \n The AutoML algorithm is run on the path " + str(self.path) + ". \n The accuracy of the model is " + str(self.accuracy) + ". \n The feature importance of the model is " + str(self.feature_importance) + "."

class Results():

    results: [Result]

    def __init__(self):
        self.results = []

    def getResults(self):
        return self.results
    
    def addResult(self, result: Result):
        self.results.append(result)
    
    def __str__(self) -> str:
        ret_str = "Results:"
        for i in self.results:
            ret_str += "\n" + str(i)
        return ret_str
    
    # def add_feature():

    # def remove_feature():
        
    
    def getBestResult(self) -> Result:
        best_result = self.results[0]
        for i in self.results:
            if i.accuracy > best_result.accuracy:
                best_result = i
        return best_result
    
    def showGraphs(self):
        for i in self.results:
            i.showGraph()
    
    def showTables(self):
        i: Result
        for i in self.results:
            i.showTable()

    def explain(self) -> str:
        return "The results are calculated by evaluating each path with the AutoML algorithm AutoGluon. \n The AutoML algorithm is run on all " + str(len(self.results)) + " paths. \n The best result is " + str(self.getBestResult().accuracy) + " with the path " + str(self.getBestResult().path) + "."
    
#TODO change to join quality.
    
class Weight():
    from_dataset: str
    to_dataset: str
    from_table: str
    to_table: str
    from_col: str
    to_col: str
    weight: float

    def __init__(self, from_dataset, to_dataset, from_table, to_table, from_col, to_col, weight):
        self.from_dataset = from_dataset
        self.to_dataset = to_dataset
        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.weight = weight

    def getFromPrefix(self):
        return self.from_dataset + "/" + self.from_table + "." + self.from_col
    
    def getToPrefix(self):
        return self.to_dataset + "/" + self.to_table + "." + self.to_col
    
    def __str__(self) -> str:
        return "Weight from " + self.getFromPrefix + "to" + self.getToPrefix + " with weight " + str(self.weight)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def why(self) -> str:
        return "This weight is calculated by the COMA algorithm. This calculates the similarity between the columns of the tables. The higher the similarity, the higher the weight. \n The weight from " + self.getFromPrefix() + " to " + self.getToPrefix() + " is " + str(self.weight) + "."

class Weights():
    weights: [Weight]
    threshold: float

    def __init__(self, threshold: float = 0.5):
        self.weights = []
        if threshold is not None:
            self.threshold = threshold
    
    def getWeights(self):
        return self.weights
    
    def getWeight(self, table_name = None, table_col = None):
        if table_name is None and table_col is None:
            return self.weights
        elif table_name is not None and table_col is None:
            return self.getWeightByTable(table_name)
        elif table_name is not None and table_col is not None:
            return self.getWeightByTableAndCol(table_name, table_col)
    
    def getWeightByTable(self, table_name: str):
        list = []
        for i in self.weights:
            if table_name == i.table1:
                list.append(i)
        return list
    
    def getWeightByFromTable(self, table_name: str) -> [Weight]:
        list = []
        for i in self.weights:
            if table_name == i.from_table:
                list.append(i)
        return list
    
    def getWeightsByFromAndToTable(self, from_table: str, to_table: str) -> [Weight]:
        list: [Weight] = []
        for i in self.weights:
            if from_table == i.from_table and to_table == i.to_table:
                list.append(i)
        # get highest weight
        return list
    
    def addWeight(self, weight: Weight):
        self.weights.append(weight)
    
    def __str__(self) -> str:
        ret_str = "Weights:"
        for i in self.weights:
            ret_str += "\n \t" + str(i)
        return ret_str
    
    
    def showWeightsByThreshold(self, threshold: float):
        list = []
        for i in self.weights:
            if i.weight > threshold:
                list.append(i)
        return list

    def getTables(self) -> list[str]:
        list = []
        for i in self.weights:
            if i.from_table not in list:
                list.append(i.from_table)
            if i.to_table not in list:
                list.append(i.to_table)
        return list

    def explain(self) -> str:
        return "The weights are calculated by the COMA algorithm. This calculates the similarity between the columns of the tables. The higher the similarity, the higher the weight. \n The algorithm is run on all combinations of " + str(len(self.getTables())) + " tables. \n The algorithm has found " + str(len(self.weights)) + " weights. \n The threshold for the similarity of the weights is " + str(self.threshold) + "."

# This class functions as loading in datasets and calculating the COMA score between each table in the datasets.
class AutofeatClass:
    targetColumn: str
    threshold: float
    base_table: str
    paths: Paths
    weights: Weights
    results: Results
    base_dataset: str
    partial_join: pd.DataFrame
    extra_tables: [(str, str)]
    exlude_tables: [(str, str)]
    partial_join_selected_features: dict = {}
    explore: bool
    data_quality_threshhold: float

    def __init__(self, targetColumn: str, base_table: str, base_dataset: str, threshold: float = 0.5, explore=False, data_quality_threshhold: float = 0.8):
        self.base_dataset = base_dataset
        self.weights = Weights(threshold=threshold)
        self.results = Results()
        self.discovered: Set[str] = set()
        self.targetColumn = targetColumn
        self.base_table = base_table
        X_train, X_test = train_test_split(get_df_with_prefix(self.base_dataset, self.base_table, self.targetColumn), random_state=42)
        self.partial_join = X_train.copy()
        features = list(X_train.columns)
        features.remove(targetColumn)
        self.partial_join_selected_features[base_table] = features 
        self.extra_tables = []
        self.exlude_tables = []
        self.definite_features = []
        self.exclude_features = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.explore = explore
        self.data_quality_threshhold = data_quality_threshhold
    
    # This function calculates the COMA weights between 2 tables in the datasets.
    def calculateComa(self, table1: pd.DataFrame, table2: pd.DataFrame) -> dict:
        matches = valentine_match(table1, table2, Coma())
        for m in matches.items():
            logging.debug(m)
        return matches
    
    def runAllComa(self):
        if self.explore:
            files = glob.glob("data/benchmark/**/*.csv")
        else:
            files = glob.glob("data/benchmark/" + self.base_dataset +"/*.csv")
        for i in self.extra_tables:
            files.append("data/benchmark/" + i[0] + "/" + i[1])
        for i in self.exlude_tables:
            files.remove("data/benchmark/" + i[0] + "/" + i[1])
        logging.info("Step 1: Running COMA on all " +  str(int(len(files) * (len(files) - 1) / 2)) + " combinations of tables")
        for comb in tqdm(itertools.combinations(files, 2), total=len(files) * (len(files) - 1) / 2):
            df1 = pd.read_csv(comb[0])
            df2 = pd.read_csv(comb[1])
            dataset1 = comb[0].split("/")[2]
            dataset2 = comb[1].split("/")[2]
            df1_name = comb[0].split("/")[-1]
            df2_name = comb[1].split("/")[-1]
            matches = self.calculateComa(df1, df2)
            for m in matches.items():
                ((_, col_from), (_, col_to)), similarity = m
                if similarity > self.weights.threshold:
                    self.__addWeight(dataset1, dataset2, df1_name, df2_name, col_from, col_to, similarity)
                    
        
    #This function returns the weight with optional params table 1 and table2
    def showWeights(self, table1=None, table2=None):
        if table1 is None and table2 is None:
            return self.weights
        elif table1 is not None and table2 is None:
            return self.weights[table1]
        elif table1 is None and table2 is not None:
            return self.weights[table2]
        else:
            return self.weights[table1][table2]
        
    def __addWeight(self, from_dataset, to_dataset, from_table, to_table, from_col, to_col, weight):
        self.weights.addWeight(Weight(from_dataset, to_dataset, from_table, to_table, from_col, to_col, weight))
        self.weights.addWeight(Weight(to_dataset, from_dataset, to_table, from_table, to_col, from_col, weight))
    
    def addManualWeight(self, from_table, to_table, from_col, to_col):
        self.weights.addWeight(Weight(from_table, to_table, from_col, to_col, 1))


    def addTable(self, dataset: str, table: str):
        print("This means that the algorithm has to recalculate it's weights and paths.")
        self.extra_tables.append((dataset, table))

    def exludeTable(self, dataset: str, table: str):
        print("This means that the algorithm has to recalculate it's weights and paths.")
        self.exlude_tables.append((dataset, table))

    def addFeature(self, dataset: str, table: str, feature: str):
        print("This means that the algorithm has to recalculate it's weights and paths.")
        self.definite_features.append(dataset + "/" + table + "." + feature)

    # This function calculates the join paths generated by the algorithm
    def calculatepaths(self, included_nodes: list = []):
        logging.info("Step 2: Calculating paths")
        self.__stream_feature_selection(queue={self.base_table}, included_nodes=included_nodes)
        self.paths = Paths()
        Path1 = Path(self.base_table, [], 0)
        Path2 = Path(self.base_table, [Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_1.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random())], 0.9960580690271078)
        Path3 = Path(self.base_table, [Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_1.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_2.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random())], 1.0055009629544862)
        Path4 = Path(self.base_table, [Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_1.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_2.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_3.csv", from_col="Key_1_2", to_col="Key_1_1", data_quality=random.random())], 1.0018457434129047)
        Path5 = Path(self.base_table, [Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_1.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_2.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_3.csv", from_col="Key_1_2", to_col="Key_1_1", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_4.csv", from_col="Key_1_2", to_col="Key_1_1", data_quality=random.random())], 1.020314831381489)
        Path6 = Path(self.base_table, [Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_1.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_0_0.csv", to_table="credit/table_1_2.csv", from_col="Key_0_0", to_col="Key_0_0", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_3.csv", from_col="Key_1_2", to_col="Key_1_1", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_4.csv", from_col="Key_1_2", to_col="Key_1_1", data_quality=random.random()), Join(from_table="credit/table_1_2.csv", to_table="credit/table_2_5.csv", from_col="Key_1_2", to_col="Key_1_2", data_quality=random.random())], 0.8822128295535643)
        self.paths.addPaths([Path1, Path2, Path3, Path4, Path5, Path6])
        p: Path
        for p in self.paths.paths:
            j: Join
            for j in p.path:
                df = pd.read_csv("data/benchmark/" + j.to_table)
                dict = {}
                features = list(df.columns)
                for f in features:
                    rel = random.random()
                    red = random.random()
                    dict[f] = {"rel": rel, "red": red}
                j.rel_red = dict

        self.testFeatures = {'credit/table_1_1.csv.job': 0.02200000000000002, 'credit/table_1_1.csv.housing': 0.019000000000000017, 'credit/table_2_3.csv.property_magnitude': 0.01100000000000001, 'credit/table_0_0.csv.Key_0_0': 0.010000000000000009, 'credit/table_0_0.csv.employment': 0.007000000000000029, 'credit/table_1_2.csv.existing_credits': 0.006000000000000005, 'credit/table_1_2.csv.own_telephone': 0.0050000000000000044, 'credit/table_1_2.csv.Key_1_2': 0.0050000000000000044, 'credit/table_1_1.csv.Key_1_1': 0.004000000000000026, 'credit/table_1_1.csv.other_parties': 0.0010000000000000009, 'credit/table_0_0.csv.residence_since': 0.0, 'credit/table_2_3.csv.foreign_worker': 0.0, 'credit/table_2_3.csv.installment_commitment': 0.0, 'credit/table_0_0.csv.num_dependents': 0.0}
    
    def __stream_feature_selection(self, queue: set, previous_queue: set = None, included_nodes: list = []):
        if len(queue) == 0:
            return
        if previous_queue is None:
            previous_queue = queue.copy()
        all_neighbours = set()
        while len(queue) > 0:
            base_node_id = queue.pop()
            self.discovered.add(base_node_id)
            near_weights = self.__get_adjacent_nodes({base_node_id}, 0.5)
            neighbours = [n.to_table for n in near_weights]
            all_neighbours.update(neighbours)
            for n in neighbours:
                join_keys = self.weights.getWeightsByFromAndToTable(base_node_id, n)
                right_df = get_df_with_prefix(join_keys[0].to_dataset, join_keys[0].to_table)
                current_queue = set()
                while len(previous_queue) > 0:
                    previous_table_join = previous_queue.pop()
                    previous_join = None
                    if previous_table_join == self.base_table:
                        previous_join = self.partial_join.copy()
                    else:
                        pass
                    prop: Weight
                    for prop in join_keys:
                        joined_df = pd.merge(left=previous_join, right=right_df, left_on=(prop.getFromPrefix()), right_on=(prop.getToPrefix()), how="left")
                        data_quality= self.data_quality_calculation(joined_df, prop)
                        if data_quality < self.data_quality_threshhold:
                            continue
                        right_columns = list(right_df.columns)
                        df = AutoMLPipelineFeatureGenerator(
                            enable_text_special_features=False, enable_text_ngram_features=False
                        ).fit_transform(X=joined_df, random_state=42, random_seed=42)

                        X = df.drop(columns=[self.targetColumn])
                        targetColumn = df[self.targetColumn]

                        relevant_features = self.calculate_relevance(joined_df.copy(), right_columns, targetColumn)
                        redundant_features = self.calculate_redundancy(joined_df.copy(), right_columns)

    def data_quality_calculation(self, joined_df: pd.DataFrame, prop: Weight) -> float:
        total_length = joined_df.shape[0]
        non_nulls = joined_df[prop.getToPrefix()].count()
        return non_nulls / total_length

    def calculate_relevance(self, joined_df: pd.DataFrame, right_columns: list, targetColumn):
        
        features = list(set(X.columns).intersection(set(right_columns)))
        correlation = abs(pearson_correlation(np.array(X[features]), np.array(targetColumn)))
        final_scores = []
        for value, name in list(zip(correlation, features)):
            if 0 < value < 1:
                final_scores.append((name, value))
        return sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    def calculate_redundancy(self, joined_df: pd.DataFrame, right_columns: list):

    
    def getDataSetFromNode(self, node: str):
        return pd.read_csv("data/benchmark/" + node)
    
    def __get_adjacent_nodes(self, nodes: list, threshold: float) -> [Weight]:
        return_list = list()
        for n in nodes:
            for x in self.weights.getWeightByFromTable(n):
                return_list.append(x)
        return return_list
    
    def get_path_length(self, path: str) -> int:
        path_tokens = path.split("--")
        return len(path_tokens) - 1
    
    def evaluatePaths(self, top_k_paths: int = 15):
        # Why sort?
        # sorted_paths = sorted(self.paths.items(), key=lambda r: (r[1], self.get_path_length(r[0])), reverse=True)
        # top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
        logging.info("Step 3: Evaluating paths")
        for path in tqdm(self.paths.getPaths(), total=len(self.paths.getPaths())):
            self.evaluatePath(path)
        
    
    def evaluatePath(self, path: Path):
        base_df = pd.read_csv("data/benchmark/" + self.dataset + "/" + path.begin)
        for i in path.path:
            df = pd.read_csv("data/benchmark/" + i.to_table)
            base_df = pd.merge(base_df, df, left_on=i.from_col, right_on=i.to_col, how="left")
        X_train, X_test, y_train, y_test = train_test_split(base_df.drop(columns=[self.targetColumn]), base_df[self.targetColumn], test_size=0.2, random_state=10)
        X_train[self.targetColumn] = y_train
        X_test[self.targetColumn] = y_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = TabularPredictor(label=self.targetColumn,
                                        problem_type="binary",
                                    verbosity=0,
                                    path= "AutogluonModels/" + "models").fit(train_data=X_train,
                                                                        hyperparameters={'LR': {'penalty': 'L1'}})
        model_names = predictor.model_names()
        for model in model_names:
            result = Result()
            res = predictor.evaluate(X_test, model=model)
            result.setAccuracy(res['accuracy'])
            ft_imp = predictor.feature_importance(data=X_test, model=model, feature_stage="original")
            result.feature_importance = dict(zip(list(ft_imp.index), ft_imp["importance"])),
            result.model = model
            result.rank = path.rank
            result.path = path
            result.setData(base_df)
            self.results.addResult(result)

    def setDatasets(self, data: []):
        self.datasets = pd.DataFrame(data)
    
    def runAlgorithm(self):
        self.runAllComa()
        self.calculatepaths()
        self.evaluatePaths()

# main run
if __name__ == "__main__":
    autofeat = AutofeatClass(targetColumn="class", base_dataset= "credit", base_table = "table_0_0.csv", threshold = 0.5, explore=False)
    autofeat.runAllComa()
    autofeat.calculatepaths()
    # autofeat.addTable("school", "qr.csv")
    # autofeat.addFeature("school", "qr.csv", "qr")
    # autofeat.runAlgorithm()
    # print(autofeat.weights)
    # print(autofeat.paths)
    # autofeat.results.results[8].showGraph()
