from sklearn.ensemble import RandomForestClassifier
from phermes import Problem
from phermes import HyperHeuristic
from typing import List
import pandas as pd

class RF(HyperHeuristic):

	def __init__(self, features : List[str], heuristics : List[str], params : dict):
		super().__init__(features, heuristics)
		self._model = None
		self._params = params

	def train(self, filename : str) -> None:
		data =  pd.read_csv(filename, header = 0)		
		columns = ["INSTANCE", "BEST", "ORACLE"] + self._heuristics 		
		X = data.drop(columns, axis = 1).values		
		y = data["BEST"].values
		for i in range(len(self._heuristics)):			
			y[y == self._heuristics[i]] = i		
		y = y.astype("int")
		self._model = RandomForestClassifier(**self._params)
		self._model.fit(X, y)

	def getHeuristic(self, problem : Problem) -> str:
		state = pd.DataFrame()
		for i in range(len(self._features)):
			state[self._features[i]] = [problem.getFeature(self._features[i])]				
		prediction = self._model.predict(state.values)
		return self._heuristics[prediction[0]]