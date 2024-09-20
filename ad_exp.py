import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model,load_model
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import pickle
import shap

input_dir = "Input/AE/"
input_data_dir = input_dir + "Data/"
input_model_dir = input_dir + "Model/"

output_dir = "Output/AE/"

class ReconstructionErrorModel():
	model = None
	dr_type = None

	def __init__(self, dr_type, model):

		self.dr_type = dr_type
		self.model = model

	def predict(self, input_data):
		reconstructed_data = self.__reconstruct_data(input_data)		
		errors = np.mean((input_data - reconstructed_data)**2, axis=1)
		return np.asarray(errors)
	
	def __reconstruct_data(self, data):
		reconstructed_data = None
	
		if dr_type == "PCA" or dr_type == "KPCA" or dr_type == "SPCA":
			transformed_data = self.model.transform(data)  
			reconstructed_data = self.model.inverse_transform(transformed_data)
			pass
		elif dr_type == "AE":
			reconstructed_data = self.model.predict(data)
			
		return reconstructed_data

def load_test_data():

	test_data = None
	normal_test_data = None
	anomalous_test_data = None
	
	test_data = pd.read_csv(input_data_dir + "Test.csv")
	features = list(test_data.columns)
	features.remove("Label")
	
	normal_test_data = test_data.loc[test_data['Label'] == "N"].drop(["Label"], axis=1)
	anomalous_test_data = test_data.loc[test_data['Label'] == "A"].drop(["Label"], axis=1)
	
	return normal_test_data, anomalous_test_data, features

def load_model(dr_type):

	model = None
	threshold = 0.0

	if dr_type == "PCA" or dr_type == "KPCA" or dr_type == "SPCA":
		model = pickle.load(open(input_model_dir + dr_type + ".pkl", 'rb'))
		file = open(input_model_dir + "threshold.txt", "r")
		threshold = float(file.readline())
		file.close()
	elif dr_type == "AE":
		model = tf.keras.models.load_model(input_model_dir + "AE.keras")
		file = open(input_model_dir + "threshold.txt", "r")
		threshold = float(file.readline())
		file.close()

	return model, threshold
	
def get_factor_effects(normal_test_data, anomalous_test_data, model, features):

	normal_shap_values = None
	anomalous_shap_values = None

	error_model = ReconstructionErrorModel(dr_type, model)
	explainer = shap.KernelExplainer(error_model.predict, normal_test_data)
	#normal_shap_values = explainer.shap_values(normal_test_data)
	normal_shap_values = explainer.shap_values(shap.kmeans(normal_test_data,25).data)
	normal_shap_values = [abs(i) for i in normal_shap_values]
	normal_shap_values = pd.DataFrame(data = normal_shap_values, columns = features)
	normal_shap_values = normal_shap_values.mean()
	normal_shap_values = pd.DataFrame(data = [list(normal_shap_values)], columns = features)
	
	explainer = shap.KernelExplainer(error_model.predict, anomalous_test_data)
	#anomalous_shap_values = explainer.shap_values(anomalous_test_data)
	anomalous_shap_values = explainer.shap_values(shap.kmeans(anomalous_test_data,25).data)
	anomalous_shap_values = [abs(i) for i in anomalous_shap_values]
	anomalous_shap_values = pd.DataFrame(data = anomalous_shap_values, columns = features)
	anomalous_shap_values = anomalous_shap_values.mean()
	anomalous_shap_values = pd.DataFrame(data = [list(anomalous_shap_values)], columns = features)

	return normal_shap_values, anomalous_shap_values
	
def save_shap_values(normal_shap_values, anomalous_shap_values):

	normal_shap_values.to_csv(output_dir + "SHAP_N.csv", index = False)
	anomalous_shap_values.to_csv(output_dir + "SHAP_A.csv", index = False)

	return None
	
try:
	dr_type = sys.argv[1]

except:
	print("Enter the right number of input arguments.")
	sys.exit()
	
normal_test_data, anomalous_test_data, features = load_test_data()
model, threshold = load_model(dr_type)
normal_shap_values, anomalous_shap_values = get_factor_effects(normal_test_data, anomalous_test_data, model, features)
save_shap_values(normal_shap_values, anomalous_shap_values)




