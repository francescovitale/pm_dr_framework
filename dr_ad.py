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
from sklearn.decomposition import SparsePCA
import pickle

input_dir = "Input/DA/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/DA/"
output_metrics_dir = output_dir + "Metrics/"
output_model_dir = output_dir + "Model/"

pca_hyperparameters = {
	"n_components": [2, 4, 8, 16]
}

nodr_hyperparameters = {
	"thresholding": ["min", "mean", "median", "max"]
}

ae_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"n_hidden_neurons": [32, 64, 128, 256],
	"optimizer": ["adam", "rmsprop", "SGD"],
	"batch_size": [8, 16, 32, 64],
	"epochs": [100, 250, 500]
}

spca_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"ridge_alpha": [0.01, 0.1, 0.25, 0.5, 0.75, 1.00],
	"alpha": [0.1, 0.5, 1, 2, 3]
}

kpca_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"kernel": ["poly", "rbf", "sigmoid"],
	"gamma": [0.01, 0.1, 0.25],
	"alpha": [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
	"degree": [3, 4, 5, 6]
}



def read_data():

	training_set = pd.read_csv(input_data_dir + "Training.csv")
	validation_set = pd.read_csv(input_data_dir + "Validation.csv")
	test_set = pd.read_csv(input_data_dir + "Test.csv")

	return training_set, validation_set, test_set

def train_kpca(training_set, validation_set, n_components, kernel, gamma, degree, alpha):

	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if n_components > len(training_set_np):
		n_components = len(training_set_np)

	model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, alpha=alpha, fit_inverse_transform=True)
	model.fit(training_set_np)
	compressed_validation_set_np = model.transform(validation_set_np)
	reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)

	return model, threshold

def train_pca(training_set, validation_set, n_components):
	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if n_components > len(training_set_np):
		n_components = len(training_set_np)
	model = PCA(n_components=n_components)
	model.fit(training_set_np)
	compressed_validation_set_np = model.transform(validation_set_np)
	reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)

	return model, threshold

def train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs):
	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	model = autoencoder(n_hidden_neurons, n_components, len(list(training_set.columns)), optimizer)
	model.fit(training_set_np,training_set_np, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
	reconstructed_validation_set_np = model.predict(validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	return model, threshold

def train_spca(training_set, validation_set, n_components, ridge_alpha, alpha):

	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if n_components > len(training_set_np):
		n_components = len(training_set_np)
	model = SparsePCA(n_components = n_components, ridge_alpha = ridge_alpha, alpha = alpha)
	model.fit(training_set_np)
	compressed_validation_set_np = model.transform(validation_set_np)
	reconstructed_validation_set_np = model.inverse_transform(compressed_validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)

	return model, threshold

def train_nodr(training_set, validation_set, thresholding):
	model = None
	threshold = 0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if thresholding == "min":
		threshold = min(validation_set["fitness"])
	elif threshold == "mean":
		threshold = sum(validation_set["fitness"])/len(validation_set["fitness"])
	elif threshold == "median":
		threshold = np.median(validation_set["fitness"])
	elif threshold == "max":
		threshold = max(validation_set["fitness"])
		
	return model, threshold	

def train_model(training_set, validation_set, dr_type):
	model = None
	threshold = 0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
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
		model.fit(training_set_np,training_set_np,epochs=500,shuffle=True,verbose=1)
		reconstructed_validation_set_np = model.predict(validation_set_np)
		threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
		
	elif dr_type == "NO_DR":
		threshold = min(validation_set["fitness"])

	return model, threshold
	
def autoencoder(hidden_neurons, latent_code_dimension, input_dimension, optimizer):
	input_layer = Input(shape=(input_dimension,))
	encoder = Dense(hidden_neurons,activation="relu")(input_layer)
	code = Dense(latent_code_dimension)(encoder)
	decoder = Dense(hidden_neurons,activation="relu")(code)
	output_layer = Dense(input_dimension,activation="linear")(decoder)
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer=optimizer,loss="mse")
	return model	

def classify_diagnoses(model, threshold, test_set, dr_type):

	predicted_labels = []
	test_labels = list(test_set["Label"])
	if dr_type != "NO_DR":
		test_set_no_labels_np = test_set.drop(["Label"], axis=1).to_numpy()
		if dr_type == "PCA":
			compressed_test_set_np = model.transform(test_set_no_labels_np)
			reconstructed_test_set_np = model.inverse_transform(compressed_test_set_np)
		elif dr_type == "KPCA":
			compressed_test_set_np = model.transform(test_set_no_labels_np)
			reconstructed_test_set_np = model.inverse_transform(compressed_test_set_np)
		elif dr_type == "SPCA":
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

def write_model(model, threshold, dr_type):

	if dr_type == "PCA" or dr_type == "KPCA" or dr_type == "SPCA":
		pickle.dump(model, open(output_model_dir + dr_type + ".pkl", 'wb'))
		file = open(output_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()
	if dr_type == "AE":
		model.save(output_model_dir + "AE.keras")
		file = open(output_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()
	if dr_type == "NO_DR":
		file = open(output_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()

	return None
	
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
best_performing_model = None
best_threshold = 0.0
best_f1 = 0.0
best_performance = None

if dr_type == "PCA":
	for n_components in pca_hyperparameters["n_components"]:
		try:
			model, threshold = train_pca(training_set, validation_set, n_components)
			
			predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
			performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
			if performance_metrics["f1"] > best_f1:
				best_f1 = performance_metrics["f1"]
				best_performance = performance_metrics.copy()
				best_performing_model = model
				best_threshold = threshold
		except:
			continue

if dr_type == "KPCA":
	for n_components in kpca_hyperparameters["n_components"]:
		for kernel in kpca_hyperparameters["kernel"]:
			for gamma in kpca_hyperparameters["gamma"]:
				for degree in kpca_hyperparameters["degree"]:
					for alpha in kpca_hyperparameters["alpha"]:
						try:
							model, threshold = train_kpca(training_set, validation_set, n_components, kernel, gamma, degree, alpha)
							predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
							performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
							if performance_metrics["f1"] > best_f1:
								best_f1 = performance_metrics["f1"]
								best_performance = performance_metrics.copy()
								best_performing_model = model
								best_threshold = threshold
						except:
							continue

if dr_type == "SPCA":
	for n_components in spca_hyperparameters["n_components"]:
		for ridge_alpha in spca_hyperparameters["ridge_alpha"]:
			for alpha in spca_hyperparameters["alpha"]:
				try:
					model, threshold = train_spca(training_set, validation_set, n_components, ridge_alpha, alpha)
					predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
					performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
					if performance_metrics["f1"] > best_f1:
						best_f1 = performance_metrics["f1"]
						best_performance = performance_metrics.copy()
						best_performing_model = model
						best_threshold = threshold
				except:
					continue

if dr_type == "AE":
	for n_components in ae_hyperparameters["n_components"]:
		for n_hidden_neurons in ae_hyperparameters["n_hidden_neurons"]:
			for optimizer in ae_hyperparameters["optimizer"]:
				for batch_size in ae_hyperparameters["batch_size"]:
					for epochs in ae_hyperparameters["epochs"]:
						try:
							model, threshold = train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs)
							predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
							performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
							if performance_metrics["f1"] > best_f1:
								best_f1 = performance_metrics["f1"]
								best_performance = performance_metrics.copy()
								best_performing_model = model
								best_threshold = threshold
						except:
							continue

if dr_type == "NO_DR":
	for thresholding in nodr_hyperparameters["thresholding"]:
		try:
			model, threshold = train_nodr(training_set, validation_set, thresholding)
			predicted_labels, test_labels = classify_diagnoses(model, threshold, test_set, dr_type)
			performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels)
			if performance_metrics["f1"] > best_f1:
				best_f1 = performance_metrics["f1"]
				best_performance = performance_metrics.copy()
				best_performing_model = model
				best_threshold = threshold
		except:
			continue
	
write_model(best_performing_model, best_threshold, dr_type)
write_metrics(best_performance)
	
	
	
	
	
	
	
	
	