from knnhh import KNNHH
from logreg import LOGREG
from mlpc import MLPC
from gnb import GNB
from rf import RF
from svm import SVM
from cca import CCA
from kp import KP
from bpp import BPP
from vcp import VCP
from ffp import FFP
from typing import List
from phermes import HyperHeuristic
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def characterize(domain : str, folder : str, features : List[str]) -> pd.DataFrame:
  """
  Characterizes the instances contained in a folder.
  """
  data = []
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(features) + "\r\n"
  for file in files:    
    if domain == "KP":
      problem = KP(folder + "/" + file)
    elif domain == "BPP":
      problem = BPP(folder + "/" + file)
    elif domain == "VCP":
      problem = VCP(folder + "/" + file)
    elif domain == "FFP":
      problem = FFP(folder + "/" + file)
    else:
      raise Exception("Problem domain '" + domain + "' is not recognized by the system.") 
    text += file + "\t"
    row = {"INSTANCE": file}
    for f in features:
      feature_value = round(problem.getFeature(f), 3)
      text += str(feature_value) + "\t"
      row[f] = feature_value
    text += "\r\n"
    data.append(row)  
  # print(text)
  df = pd.DataFrame(data)
  return df

def solve(domain : str, folder : str, heuristics : List[str]) -> pd.DataFrame:
  data = []
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(heuristics) + "\r\n"  
  for i in range(len(files)):
    text += files[i] + "\t"
    row = {"INSTANCE": files[i]}   
    for h in heuristics:
      if domain == "KP":
        problem = KP(folder + "/" + files[i])
      elif domain == "BPP":
        problem = BPP(folder + "/" + files[i])
      elif domain == "VCP":
      	problem = VCP(folder + "/" + files[i])
      elif domain == "FFP":
      	np.random.seed(i)
      	problem = FFP(folder + "/" + files[i])
      else:
        raise Exception("Problem domain '" + domain + "' is not recognized by the system.")      
      problem.solve(h)
      solution_value = round(problem.getObjValue(), 3)
      text += str(solution_value) + "\t"
      row[h] = solution_value
    text += "\r\n"
    data.append(row)  
  # print(text)
  df = pd.DataFrame(data)
  return df

def solveHH(domain : str, folder : str, hyperHeuristic : HyperHeuristic, hyperHeuristic_name : str, mode : str):
  data = []
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(str(hyperHeuristic_name)) + "\r\n"  
  for file in files:      
    if domain == "KP":
      problem = KP(folder + "/" + file)
    elif domain == "BPP":
      problem = BPP(folder + "/" + file)
    elif domain == "VCP":
      problem = VCP(folder + "/" + file)
    elif domain == "FFP":
      problem = FFP(folder + "/" + file)  
    else:
      raise Exception("Problem domain '" + domain + "' is not recognized by the system.")     
    text += file + "\t" 
    row = {"INSTANCE": file}    
    problem.solveHH(hyperHeuristic, mode)
    value = round(problem.getObjValue(), 3)
    text += str(value) + "\r\n"
    row[str(hyperHeuristic_name)] = value
    data.append(row)
  # print(text)
  df = pd.DataFrame(data)
  return df


def plot_heatmaps(ca_grid, heuristics, m, params_type, mode, output_path):
    n_classes = ca_grid.shape[0]  # The first dimension represents classes
    # Set up the plot figure with subplots
    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes * 5, 5))
    for i in range(n_classes):
        # Select the data for the current class
        ca_grid_2d = ca_grid[i]
        # Plot the heatmap for the current class
        ax = axes[i] if n_classes > 1 else axes
        cax = ax.imshow(ca_grid_2d, cmap='gray', interpolation='nearest')
        ax.set_title(heuristics[i])
        # Set the ticks manually
        ax.set_xticks(range(ca_grid_2d.shape[1]))
        ax.set_yticks(range(ca_grid_2d.shape[0]))
        # Set tick labels
        ax.set_xticklabels([str(i + 1) for i in range(m)])
        ax.set_yticklabels(['W', 'P', 'C'])
        fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path + f'/cca_train_heatmap_{params_type}_{mode}')
    # plt.show()

# ##############################################################################
# #################### Complementary Processing Functions ######################
    
def characterize_and_export(domain : str, features : List[str], heuristics : List[str], data_raw_path : str, export_path : str, csv_filename : str):
    """
    Characterizes and solves problem instances, combines the results, and exports to a CSV file.
    """
    df_characterize = characterize(domain, data_raw_path, features)
    # print('--------------------------------END--------------------------------')
    df_solve = solve(domain, data_raw_path, heuristics)
    # print('--------------------------------END--------------------------------')
    df_merged = pd.merge(df_characterize, df_solve, on="INSTANCE", how="outer")
    heuristic_columns = df_merged[heuristics]
    df_merged['BEST'] = heuristic_columns.idxmax(axis=1)
    df_merged['ORACLE'] = heuristic_columns.max(axis=1)
    export_path = export_path + csv_filename
    df_merged.to_csv(export_path, index=False)

def train_and_evaluate(model_class, model_name : str, params: dict, params_type : str, features : List[str], heuristics : List[str], mode : str, train_kpi_characterized_path : str, test_kpi_raw_path : str):
    model_params = params[model_name][params_type]
    model = model_class(features, heuristics, model_params)
    model.train(train_kpi_characterized_path)
    results = solveHH("KP", test_kpi_raw_path, model, model_name, mode)
    if model == CCA:
      ca_grid = model.get_ca_grid()
      plot_heatmaps(ca_grid, heuristics, params_type, mode , params[model_name][params_type]['m'])
    return results

def report_all_experiments_results(model_info : List[tuple], params: dict, params_set_type : List[str], features : List[str], heuristics : List[str], modes : List[str], test_kpi_characterized_path : str, train_kpi_characterized_path : str, test_kpi_raw_path : str, results_output_path : str):
  df_characterize = pd.read_csv(test_kpi_characterized_path)
  for mode in modes:
    for params_type in params_set_type:
      results = {}
      all_results = df_characterize.copy()
      for model_name, model_class in model_info:
        results[f'results_{model_name.lower()}_{params_type}_{mode}'] = train_and_evaluate(model_class, model_name, params, params_type, features, heuristics, mode, train_kpi_characterized_path, test_kpi_raw_path)     
      for key, df in results.items():
        all_results = pd.merge(all_results, df, on="INSTANCE", how="outer")
      all_results.set_index('INSTANCE', inplace=True)
      columns_to_drop = ["WEIGHT", "PROFIT", "CORRELATION", "BEST", "ORACLE"]
      all_results.drop(columns=columns_to_drop, inplace=True, errors='ignore')
      # Boxploting
      column_names = all_results.columns.tolist()
      plt.figure(figsize=(10, 6))
      plt.boxplot(all_results.values, labels=column_names)
      plt.title(f'Comparison of Selection HH Learning Mechanism: {params_type} parameters in {mode} behavior')
      plt.ylabel('KP Instances Profit')
      plt.xlabel('Models')
      plt.xticks(rotation=45)
      plt.grid(True)
      plt.savefig(f"{results_output_path}/comparison_{params_type}_{mode}.png")
      plt.close()
      # CSV reporting
      all_results.to_csv(f"{results_output_path}/comparison_{params_type}_{mode}.csv")
      print(f"Results for {params_type}_{mode} have been successfully saved.")
  
# ##############################################################################
# #########################  Features & Heuristics (Names)######################
features = ["WEIGHT", "PROFIT", "CORRELATION"]
heuristics = ["DEF", "MAXP", "MAXPW", "MINW"]

# ##############################################################################
# ##################################  Files Path  ##############################
kpi_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Instances"
train_kpi_raw_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Instances\Train"
test_kpi_raw_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Instances\Test"
train_kpi_characterized_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Instances\train_kpi_characterized.csv"
test_kpi_characterized_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Instances\test_kpi_characterized.csv"
# ##############################################################################
# ###############################  Results Path  ###############################
results_output_path = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Confirmatory_Experiments\Results\PHERMES_results"

# ##############################################################################
# ##############################  Characterize KPIs ############################
# characterize_and_export('KP', features, heuristics, train_kpi_raw_path, kpi_path, '\\train_kpi_characterized.csv')
# characterize_and_export('KP', features, heuristics, test_kpi_raw_path, kpi_path, '\\test_kpi_characterized.csv')

# ##############################################################################
# ##############################  Model Parameters #############################
params = {
    "CCA": {
        "custom": {'n_classes': 4, 'n_attributes': 3, 'm': 5, 'portion': 0.2, 'range_param': 0.2},
        "default": {'n_classes': 4, 'n_attributes': 3, 'm': 5, 'portion': 0.2, 'range_param': 0.2}
    },
    "KNN": {
        "custom": 3,  # 'k' renamed to 'neighbors' for clarity and consistency
        "default": 5
    },
    "LOGREG": {
        "custom": {"C": 0.8, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5, 'random_state': 42},
        "default": {'random_state': 42}
    },
    "MLPC": {
        "custom": {"hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", "alpha": 0.0001, 'random_state': 42, 'max_iter': 2500},
        "default": {'random_state': 42}
    },
    "RF": {
        "custom": {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'bootstrap': True, 'oob_score': False, 'n_jobs': None, 'random_state': 42, 'verbose': 0, 'warm_start': False, 'class_weight': None},
        "default": {'random_state': 42}
    },
    "SVM": {
        "custom": {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'scale', 'coef0': 0.0, 'shrinking': True, 'probability': False, 'tol': 1e-3, 'cache_size': 200, 'class_weight': None, 'verbose': False, 'max_iter': -1, 'decision_function_shape': 'ovr', 'break_ties': False, 'random_state': 42},
        "default": {'random_state': 42}
    }
}

model_info = [
    ('CCA', CCA),
    ('KNN', KNNHH),
    ('LOGREG', LOGREG),
    ('MLPC', MLPC),
    ('RF', RF),
    ('SVM', SVM)
]

# All models need to have same param_sets CCA size
params_set_type = list(params['CCA'].keys())

modes = ['dynamic', 'static'] 

# ##############################################################################
# ######################### Train & Evaluate Execution #########################

# ##############################################################################
# ######################### Report Models Execution ############################
report_all_experiments_results(model_info, params, params_set_type, features, heuristics, modes, test_kpi_characterized_path, train_kpi_characterized_path, test_kpi_raw_path, results_output_path)