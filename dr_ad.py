import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import pandas as pd
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import pickle

input_dir = "Input/DA/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/DA/"
output_metrics_dir = output_dir + "Metrics/"


def read_data():

	training_set = pd.read_csv(input_data_dir + "Training.csv")
	validation_set = pd.read_csv(input_data_dir + "Validation.csv")
	test_set = pd.read_csv(input_data_dir + "Test.csv")

	return training_set, validation_set, test_set

def train_model(training_set, validation_set, dr_type):
	model = None
	threshold = 0.0
	training_set_np = training_set.to_numpy()
	#training_set_np = np.delete(training_set_np, 0, 0)
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	#validation_set_np = np.delete(validation_set_np, 0, 0)
	validation_set_np = validation_set_np.astype('float32')
	
	if int(len(list(training_set.columns))/4) < len(training_set_np):
		n_components = int(len(list(training_set.columns))/4)
	else:
		n_components = len(training_set_np)
		
	if dr_type == "PCA":
		model = PCA(n_components=n_components)
		model.fit(training_set_np)
		compressed_validation_set_np = model.transform(validation_set_np)
		reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
		threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	elif dr_type == "KPCA":
		model = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.1, alpha=0.01, fit_inverse_transform=True)
		model.fit(training_set_np)
		compressed_validation_set_np = model.transform(validation_set_np)
		reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
		threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	elif dr_type == "AE":
		model = autoencoder(int(len(list(training_set.columns))/2), int(len(list(training_set.columns))/4), len(list(training_set.columns)))
		model.fit(training_set_np,training_set_np,epochs=250,shuffle=True,verbose=1)
		reconstructed_validation_set_np = model.predict(validation_set_np)
		threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	elif dr_type == "NO_DR":
		threshold = min(validation_set["fitness"])

	return model, threshold
	
def autoencoder(hidden_neurons, latent_code_dimension, input_dimension):
	input_layer = Input(shape=(input_dimension,))
	encoder = Dense(hidden_neurons,activation="relu")(input_layer)
	code = Dense(latent_code_dimension)(encoder)
	decoder = Dense(hidden_neurons,activation="relu")(code)
	output_layer = Dense(input_dimension,activation="linear")(decoder)
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer="adam",loss="mse")
	return model	

def classify_diagnoses(model, threshold, test_set, dr_type):

	predicted_labels = []
	test_labels = list(test_set["Label"])
	if dr_type != "NO_DR":
		test_set_no_labels_np = test_set.drop(["Label"], axis=1).to_numpy()
		#test_set_no_labels_np = np.delete(test_set_no_labels_np, 0, 0)
		if dr_type == "PCA":
			compressed_test_set_np = model.transform(test_set_no_labels_np)
			reconstructed_test_set_np = model.inverse_transform(compressed_test_set_np)
		elif dr_type == "KPCA":
			compressed_test_set_np = model.transform(test_set_no_labels_np)
			reconstructed_test_set_np = model.inverse_transform(compressed_test_set_np)
		elif dr_type == "AE":
			reconstructed_test_set_np = model.predict(test_set_no_labels_np, verbose=0)
		
		for idx,elem in enumerate(reconstructed_test_set_np):
			error = mean_squared_error(test_set_no_labels_np[idx], reconstructed_test_set_np[idx])
			if error > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")
	else:
		for idx, row in test_set.iterrows():
			if row["fitness"] >= threshold:
				predicted_labels.append("N")
			else:
				predicted_labels.append("A")

	return predicted_labels, test_labels

def evaluate_performance_metrics(test_labels, predicted_labels):
	performance_metrics = {}
	
	performance_metrics["accuracy"] = accuracy_score(test_labels, predicted_labels)
	performance_metrics["precision"] = precision_score(test_labels, predicted_labels, average="macro")
	performance_metrics["recall"] = recall_score(test_labels, predicted_labels, average="macro")
	performance_metrics["f1"] = f1_score(test_labels, predicted_labels, average="macro")
	cm = confusion_matrix(test_labels, predicted_labels)
	try:
		performance_metrics["tn"] = cm[0][0]
	except:
		performance_metrics["tn"] = 0
	try:
		performance_metrics["tp"] = cm[1][1]
	except:
		performance_metrics["tp"] = 0
	try:
		performance_metrics["fn"] = cm[1][0]
	except:
		performance_metrics["fn"] = 0
	try:
		performance_metrics["fp"] = cm[0][1]
	except:
		performance_metrics["fp"] = 0
	
	print("The evaluated performance metrics are the following:")
	print("Accuracy: " + str(performance_metrics["accuracy"]))
	print("Precision: " + str(performance_metrics["precision"]))
	print("Recall: " + str(performance_metrics["recall"]))
	print("f1: " + str(performance_metrics["f1"]))
	print("TP: " + str(performance_metrics["tp"]))
	print("TN: " + str(performance_metrics["tn"]))
	print("FN: " + str(performance_metrics["fn"]))
	print("FP: " + str(performance_metrics["fp"]))
	
	
	return performance_metrics
	
def write_metrics(performance_metrics):

	file = open(output_metrics_dir + "Metrics.txt", "w")
	file.write("Accuracy: " + str(performance_metrics["accuracy"]) + "\n")
	file.write("Precision: " + str(performance_metrics["precision"]) + "\n")
	file.write("Recall: " + str(performance_metrics["recall"]) + "\n")
	file.write("f1: " + str(performance_metrics["f1"]) + "\n")
	file.write("TN: " + str(performance_metrics["tn"]) + "\n")
	file.write("TP: " + str(performance_metrics["tp"]) + "\n")
	file.write("FN: " + str(performance_metrics["fn"]) + "\n")
	file.write("FP: " + str(performance_metrics["fp"]))
	file.close()

try:
	dr_type = sys.argv[1]
	
except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()

training_set, validation_set, test_set = read_data()

model, threshold = train_model(training_set, validation_set, dr_type)
predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
write_metrics(performance_metrics)
	
	
	
	
	
	
	
	
	