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

def characterize(domain : str, folder : str, features : List[str]):
  """
  Characterizes the instances contained in a folder.
  """
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
    for f in features:
      text += str(round(problem.getFeature(f), 3)) + "\t"
    text += "\r\n"  
  print(text)

def solve(domain : str, folder : str, heuristics : List[str]):
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\t" + "\t".join(heuristics) + "\r\n"  
  for i in range(len(files)):    
    text += files[i] + "\t"   
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
      text += str(round(problem.getObjValue(), 3)) + "\t"
    text += "\r\n"  
  print(text)

def solveHH(domain : str, folder : str, hyperHeuristic : HyperHeuristic):
  results = []
  text = ""
  files = os.listdir(folder)
  files.sort()
  text += "INSTANCE\tHH\r\n"
  for file in files:    
    text += file + "\t"   
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
    problem.solveHH(hyperHeuristic)
    value = round(problem.getObjValue(), 3)
    results.append(value)
    text += str(value) + "\r\n"
  print(text)
  return results

def plot_heatmaps(ca_grid, heuristics, m):
    n_classes = ca_grid.shape[0]  # Assuming the first dimension represents classes

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
    plt.savefig(r'C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Results\PHERMES_GAF_50\heatmap.png')
    plt.show()

features = ["WEIGHT", "PROFIT", "CORRELATION"]
heuristics = ["DEF", "MAXP", "MAXPW", "MINW"]
# ################################################################################
# ###############################  Characterize KPIs #############################

# Trains and tests a KNN hyper-heuristic on any of the given problem domains.
# To test it, uncomment the corresponding code.

# characterize("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Balanced KP Instances 2024\Training", features)
# solve("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Balanced KP Instances 2024\Training", heuristics)
    
# characterize("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Balanced KP Instances 2024\Test", features)
# solve("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Balanced KP Instances 2024\Test", heuristics)    

################################################################################
####################################  KNN  #####################################

# Custom Parameters
knn = KNNHH(features, heuristics, 3)

# # Default Parameters
# knn = KNNHH(features, heuristics, 5)

knn.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_knn = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", knn)

###############################################################################
###################################  CCA  #####################################
params = {
      'n_classes': 4,
      'n_attributes': 3,
      'm': 5,  # Number of cells in each row
      'portion': 0.2,  # Portion of heat to distribute
      'range_param': 0.2  # Range parameter for heat distribution
}
cca = CCA(features, heuristics, params)

cca.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_cca = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", cca)

ca_grid = cca.get_ca_grid()
plot_heatmaps(ca_grid, heuristics, params['m'])

###############################################################################
###################################  LOGREG  #####################################

# Custom Parameters
params = {
  "C": 0.8,
  "penalty": "elasticnet",
  "solver": "saga",
  "l1_ratio": 0.5,
  'random_state': 42
}
logreg = LOGREG(features, heuristics, params)

# # Default Parameters
# logreg = LOGREG(features, heuristics, {'random_state':42})

logreg.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_logreg = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", logreg)

###############################################################################
###################################  MLPC  #####################################

# # Custom Parameters
params = {
  "hidden_layer_sizes": (100,), 
  "activation": "relu", 
  "solver": "adam", 
  "alpha": 0.0001,
  'random_state': 42,
  'max_iter': 2500
  }
mlpc = MLPC(features, heuristics, params)

# # Default Parameters
# mlpc = MLPC(features, heuristics, {'random_state':42})

mlpc.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_mlpc = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", mlpc)

###############################################################################
###################################  RF  #####################################

# # Custom Parameters
params = {
    'n_estimators': 100,
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': None,
    'random_state': 42,
    'verbose': 0,
    'warm_start': False,
    'class_weight': None,
}
rf = RF(features, heuristics, params)

# # Default Parameters
# rf = RF(features, heuristics, {'random_state':42})

rf.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_rf = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", rf)

###############################################################################
###################################  SVM  #####################################

# # Custom Parameters
params = {
    'C': 1.0,
    'kernel': 'rbf',
    'degree': 3,
    'gamma': 'scale',
    'coef0': 0.0,
    'shrinking': True,
    'probability': False,
    'tol': 1e-3,
    'cache_size': 200,
    'class_weight': None,
    'verbose': False,
    'max_iter': -1,
    'decision_function_shape': 'ovr',
    'break_ties': False,
    'random_state': 42
}
svm = SVM(features, heuristics, params)

# # Default Parameters
# svm = SVM(features, heuristics, {'random_state':42})

svm.train(r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Training-Balanced.csv")
results_svm = solveHH("KP", r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\Balanced KP Instances 2024\Test", svm)

###############################################################################
###################################  Comparison  #####################################

test = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\PHERMES\Instances\KP\KP-Test-Balanced.csv"
df = pd.read_csv(test)
insta = df['INSTANCE'].tolist()
defa = df['DEF'].tolist()
maxp = df['MAXP'].tolist()
maxpw = df['MAXPW'].tolist()
minw = df['MINW'].tolist()

all_results = [insta, defa, maxp, maxpw, minw, results_knn, results_cca, results_logreg, results_mlpc, results_rf, results_svm]
labels = ["INSTANCE", "DEF", "MAXP", "MAXPW", "MINW", "KNN", "CCA", "LOGREG", "MLPC", "RF", "SVM"]

#################################### Create boxplot ####################################
plt.figure(figsize=(10, 6))
plt.boxplot(all_results[1:], labels=labels[1:])
plt.title('Comparison of Selection HH Learning Mechanism')
plt.ylabel('KP Instances Profit')
plt.xlabel('Models')
plt.grid(True)
plt.savefig(r'C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Results\PHERMES_GAF_50\comparison.png')
plt.show()

library_names = ['GA-DEF_EASY_100', 'GA-MAXP_EASY_100', 'GA-MAXPW_EASY_100', 'GA-MINW_EASY_100']

nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))  # Adjust the size as needed
fig.suptitle('Comparison of Selection HH Learning Mechanism by Library')

for i, library_name in enumerate(library_names):
    ax = axes[i // ncols, i % ncols]  # Define the position of the current plot

    # Filter the dataframe and the results for the current library
    segment = df[df['INSTANCE'].str.contains(library_name)]
    segment_indices = segment.index.tolist()  # Get the indices of the rows in this segment
    # Prepare the data for this segment
    segment_results = [
        segment['DEF'].tolist(), 
        segment['MAXP'].tolist(), 
        segment['MAXPW'].tolist(), 
        segment['MINW'].tolist(),
        [results_knn[idx] for idx in segment_indices],  # Filter the results arrays using segment_indices
        [results_cca[idx] for idx in segment_indices],
        [results_logreg[idx] for idx in segment_indices],
        [results_mlpc[idx] for idx in segment_indices],
        [results_rf[idx] for idx in segment_indices],
        [results_svm[idx] for idx in segment_indices]
    ]
    
    # Plot the data for this segment
    ax.boxplot(segment_results, labels=labels[1:])
    ax.set_title(f'{library_name} Instances')
    ax.set_ylabel('KP Instances Profit')
    ax.set_xlabel('Models')
    ax.grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig(r'C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Results\PHERMES_GAF_50\comparison_by_library.png')

# Show the plot
plt.show()
#################################### Create boxplot ####################################

# Transpose the list of lists
transposed_data = list(zip(*all_results))

# Write to CSV
csv_file = r"C:\Users\xedua\OneDrive\Escritorio\MCC-I\Research\Thesis_Experiments\Results\PHERMES_GAF_50\comparison.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["INSTANCE", "DEF", "MAXP", "MAXPW", "MINW", "KNN", "CCA", "LOGREG", "MLPC", "RF", "SVM"])  # Column headers
    writer.writerows(transposed_data)

print(f"Data has been successfully written to {csv_file}")