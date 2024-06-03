from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
import pm4py.algo.conformance.tokenreplay as tokenreplay
import pm4py
import os
import sys
import pandas as pd
import random
from itertools import product
from itertools import permutations
import bisect
import time
import numpy as np
from sklearn.decomposition import PCA

input_dir = "Input/PF/"
input_eventlogs_dir = input_dir + "EventLogs/"
input_test_eventlogs_dir = input_eventlogs_dir + "Test/"
input_training_eventlogs_dir = input_eventlogs_dir + "Training/"
input_validation_eventlogs_dir = input_eventlogs_dir + "Validation/"
input_normativemodel_dir = input_dir + "NormativeModel/"

output_dir = "Output/PF/"
output_data_dir = output_dir + "Data/"

normalization_technique = "zscore"

def read_normative_model():

	normative_model = {}
	for normative_model_filename in os.listdir(input_normativemodel_dir):
		normative_model["network"], normative_model["initial_marking"], normative_model["final_marking"] = pnml_importer.apply(input_normativemodel_dir + normative_model_filename)
			
	return normative_model
	
def read_event_logs():
	
	training_event_logs = []
	validation_event_logs = []
	normal_test_event_logs = []
	anomalous_test_event_logs = []
	
	for filename in os.listdir(input_training_eventlogs_dir):
		training_event_logs.append(xes_importer.apply(input_training_eventlogs_dir + filename))
		
	for filename in os.listdir(input_validation_eventlogs_dir):
		validation_event_logs.append(xes_importer.apply(input_validation_eventlogs_dir + filename))
		
	for filename in os.listdir(input_test_eventlogs_dir):
		if filename.split(".xes")[0].split("_")[1] == "A":
			anomalous_test_event_logs.append(xes_importer.apply(input_test_eventlogs_dir + filename))
		elif filename.split(".xes")[0].split("_")[1] == "N":
			normal_test_event_logs.append(xes_importer.apply(input_test_eventlogs_dir + filename))			
	
	return training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs
		
def feature_extraction(training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs, normative_model, fe_type):


	training_set = None
	validation_set = None
	test_set = None
	if fe_type == "n_grams":
		training_set = get_event_logs_ngrams(training_event_logs, 2)
		validation_set = get_event_logs_ngrams(validation_event_logs, 2)
		normal_test_set = get_event_logs_ngrams(normal_test_event_logs, 2)
		normal_test_set.insert(len(normal_test_set.columns), "Label", ["N"]*len(normal_test_set), True)
		anomalous_test_set = get_event_logs_ngrams(anomalous_test_event_logs, 2)
		anomalous_test_set.insert(len(anomalous_test_set.columns), "Label", ["A"]*len(anomalous_test_set), True)
		test_set = pd.concat([normal_test_set, anomalous_test_set], ignore_index=True)
	elif fe_type == "directly_follows":
		training_set = get_df_relationships(training_event_logs)
		validation_set = get_df_relationships(validation_event_logs)
		normal_test_set = get_df_relationships(normal_test_event_logs)
		normal_test_set.insert(len(normal_test_set.columns), "Label", ["N"]*len(normal_test_set), True)
		anomalous_test_set = get_df_relationships(anomalous_test_event_logs)
		anomalous_test_set.insert(len(anomalous_test_set.columns), "Label", ["A"]*len(anomalous_test_set), True)
		test_set = pd.concat([normal_test_set, anomalous_test_set], ignore_index=True)
	elif fe_type == "token_based_cc":
		training_set = generate_tb_diagnoses(training_event_logs, normative_model)
		validation_set = generate_tb_diagnoses(validation_event_logs, normative_model)
		normal_test_set = generate_tb_diagnoses(normal_test_event_logs, normative_model)
		normal_test_set.insert(len(normal_test_set.columns), "Label", ["N"]*len(normal_test_set), True)
		anomalous_test_set = generate_tb_diagnoses(anomalous_test_event_logs, normative_model)
		anomalous_test_set.insert(len(anomalous_test_set.columns), "Label", ["A"]*len(anomalous_test_set), True)
		test_set = pd.concat([normal_test_set, anomalous_test_set], ignore_index=True)
	elif fe_type == "alignment_based_cc":
		training_set = generate_ab_diagnoses(training_event_logs, normative_model)
		validation_set = generate_ab_diagnoses(validation_event_logs, normative_model)
		normal_test_set = generate_ab_diagnoses(normal_test_event_logs, normative_model)
		normal_test_set.insert(len(normal_test_set.columns), "Label", ["N"]*len(normal_test_set), True)
		anomalous_test_set = generate_ab_diagnoses(anomalous_test_event_logs, normative_model)
		anomalous_test_set.insert(len(anomalous_test_set.columns), "Label", ["A"]*len(anomalous_test_set), True)
		test_set = pd.concat([normal_test_set, anomalous_test_set], ignore_index=True)

	training_set.fillna(0, inplace=True)
	validation_set.fillna(0, inplace=True)
	test_set.fillna(0, inplace=True)

	training_set_columns = list(training_set.columns)
	validation_set_columns = list(validation_set.columns)
	test_set_columns = list(test_set.drop(["Label"],axis=1).columns)
	shared_columns = [x for x in training_set_columns if x in validation_set_columns]
	shared_columns = [x for x in shared_columns if x in test_set_columns]
	not_shared_training_set_columns = [x for x in training_set_columns if x not in shared_columns]
	not_shared_validation_set_columns = [x for x in validation_set_columns if x not in shared_columns]
	not_shared_test_set_columns = [x for x in test_set_columns if x not in shared_columns]
	training_set = training_set.drop(not_shared_training_set_columns, axis=1)
	validation_set = validation_set.drop(not_shared_validation_set_columns, axis=1)
	test_set = test_set.drop(not_shared_test_set_columns, axis=1)

	training_set = training_set.reindex(sorted(training_set.columns), axis=1)
	validation_set = validation_set.reindex(sorted(validation_set.columns), axis=1)
	temp = test_set.drop("Label", axis=1)
	temp = temp.reindex(sorted(temp.columns), axis=1)
	test_set = pd.concat([temp, test_set["Label"]], axis=1)
	
	return training_set, validation_set, test_set
	
def get_event_logs_ngrams(event_logs, n):

	possible_n_grams = {}
	activities = get_activities(event_logs)
	temp = list(product(activities,repeat=n))
	for n_gram in temp:
		possible_n_grams[n_gram] = 0
	per_event_log_traces = get_per_event_log_traces(event_logs)
	possible_n_grams = compute_ngrams(possible_n_grams, per_event_log_traces, n)
	
	return possible_n_grams	
	
def get_activities(event_logs):
	
	activities = []
	for event_log in event_logs:
		for trace in event_log:
			for event in trace:
				if event["concept:name"] not in activities:
					activities.append(event["concept:name"])	
	activites = list(set(activities))
	
	return activities	

def get_per_event_log_traces(event_logs):

	per_event_log_traces = []
	for event_log in event_logs:
		traces = []
		for trace in event_log:
			events = []
			for event in trace:
				events.append(event["concept:name"])
			traces.append(events)
		per_event_log_traces.append(traces)
			
	return per_event_log_traces
	
def compute_ngrams(possible_n_grams, per_event_log_traces, n):
	rows = []
	for event_log in per_event_log_traces:
		per_event_log_row = {}
		for key in possible_n_grams:
			per_event_log_row[key] = 0
		for trace in event_log:
			if len(trace) < n:
				pass
			else:
				for i in range(n-1, len(trace), n):
					current_n_gram = trace[i-n+1:i+1]
					per_event_log_row[tuple(current_n_gram)] = per_event_log_row[tuple(current_n_gram)] + 1
		rows.append(list(per_event_log_row.values()))
	
	temp = pd.DataFrame(columns = list(possible_n_grams.keys()), data = rows)
		
	return temp	
	
def get_df_relationships(event_logs):

	activities = get_activities(event_logs)
	possible_df_relationships = list(permutations(activities,2))
	possible_df_relationships = sorted(possible_df_relationships)
	rows = []
	for event_log in event_logs:
		row = {}
		for df_relationship in possible_df_relationships:
			row[df_relationship] = 0
		event_log_df = footprints_discovery.apply(event_log, variant=footprints_discovery.Variants.TRACE_BY_TRACE)
		for trace_analysis in event_log_df:
			for df_relationship in trace_analysis["sequence"]:
				row[df_relationship] = row[df_relationship] + 1
		rows.append(list(row.values()))
	possible_df_relationships = pd.DataFrame(columns=possible_df_relationships, data = rows)	
	
	return possible_df_relationships		

def generate_tb_diagnoses(event_logs, normative_model):
		
	diagnoses = {}
	columns = []

	# get the list of model activities	
	model_activities = get_model_activities(normative_model["network"].transitions)
	model_diagnoses_columns = model_activities
	model_diagnoses_columns.sort()
	model_diagnoses_columns.append("m")
	model_diagnoses_columns.append("r")
	model_diagnoses_columns.append("fitness")
	
	model_diagnoses = []
	
	# get activated transitions, missing and remaining tokens, and fitness values
	for event_log in event_logs:
		row = {}
		for column in model_diagnoses_columns:
			row[column] = 0
		row["m"] = 0
		row["r"] = 0
		row["fitness"] = 0
		for trace in event_log:
			replayed_trace = pm4py.conformance_diagnostics_token_based_replay((pm4py.objects.log.obj.EventLog)([trace]), normative_model["network"], normative_model["initial_marking"], normative_model["final_marking"])
			activated_transitions = replayed_trace[0]["activated_transitions"]
			for activated_transition in activated_transitions:
				if activated_transition.label is not None and activated_transition.label in model_diagnoses_columns:
					row[activated_transition.label] = row[activated_transition.label] + 1
			row["m"] = row["m"] + replayed_trace[0]["missing_tokens"]
			row["r"] = row["r"] + replayed_trace[0]["remaining_tokens"]		
			row["fitness"] = row["fitness"] + replayed_trace[0]["trace_fitness"]
		row["fitness"] = row["fitness"]/len(event_log)
		
		model_diagnoses.append(list(row.values()))

	# encode activated transitions, missing and remaining tokens, and fitness values as tabular data			
	model_diagnoses = pd.DataFrame(columns = model_diagnoses_columns, data = model_diagnoses)	

	return model_diagnoses	
		
def generate_ab_diagnoses(event_logs, normative_model):

	diagnoses = {}
	columns = []
	
	# get the list of model & log activities
	model_activities = get_model_activities(normative_model["network"].transitions)
	model_diagnoses_columns = []
	model_diagnoses_columns = get_model_logs_activities(model_activities.copy(), event_logs)
	model_activities = model_diagnoses_columns.copy()
	model_diagnoses_columns.sort()
	
	model_misalignments = []
	model_fitness_diagnoses = []
		
	# get model misalignments and fitness values
	for event_log in event_logs:
		log = event_log
		try:
			fitness, aligned_traces = compute_fitness(normative_model, log, "ALIGNMENT_BASED")
			misaligned_activities = compute_misaligned_activities(log, aligned_traces)
			for activity in model_activities:
				if activity not in list(misaligned_activities.keys()):
					misaligned_activities[activity] = 0
			temp_list = []
			for sorted_key in sorted(misaligned_activities):
				temp_list.append(misaligned_activities[sorted_key])
			model_misalignments.append(temp_list)
		except Exception as ex:
			fitness, ignore = compute_fitness(normative_model, log, "TOKEN_BASED")
		model_fitness_diagnoses.append(fitness)
	
	# encode model_misalignments, model_fitness_diagnoses as tabular data		
	model_diagnoses = []
	if len(model_misalignments) > 0:
		for idx,el in enumerate(model_fitness_diagnoses):
			model_diagnoses.append(model_misalignments[idx] + [model_fitness_diagnoses[idx]])	
		model_diagnoses_columns.append("fitness")
		model_diagnoses = pd.DataFrame(columns = model_diagnoses_columns, data = model_diagnoses)
			
	else:
		for idx,el in enumerate(model_fitness_diagnoses):
			model_diagnoses.append([model_fitness_diagnoses[idx]] + [model_labels[idx]])
		model_diagnoses = pd.DataFrame(columns = ["fitness"], data = model_diagnoses)
	
	return model_diagnoses		

def get_model_logs_activities(model_activities, event_logs):

	model_logs_activities = model_activities
	
	for event_log in event_logs:
		event_log_activities = []
		for trace in event_log:
			for event in trace:
				if event["concept:name"] not in event_log_activities:
					event_log_activities.append(event["concept:name"])
		
		for activity in event_log_activities:
			if activity not in model_logs_activities:
				model_logs_activities.append(activity)

	return model_logs_activities

def get_model_activities(transitions):
	
	activities = []
	for transition in transitions:
		if transition._Transition__get_label() != None:
			activities.append(transition._Transition__get_label())
	activites = list(set(activities))

	return activities
		
def compute_fitness(petri_net, event_log, cc_variant):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	if cc_variant == "ALIGNMENT_BASED":
		aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
		log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	elif cc_variant == "TOKEN_BASED":
		replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["network"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
		log_fitness = replay_fitness.evaluate(results = replay_results, variant = replay_fitness.Variants.TOKEN_BASED)["log_fitness"]



	return log_fitness, aligned_traces
	
def compute_misaligned_activities(event_log, aligned_traces):
	
	misaligned_activities = {}
	events = {}
	temp = []
	for aligned_trace in aligned_traces:
		temp.append(list(aligned_trace.values())[0])
	aligned_traces = temp
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
						events[log_behavior] = events[log_behavior]+1
				elif model_behavior != None and model_behavior != ">>":
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
						events[model_behavior] = events[model_behavior]+1
	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			misaligned_activities[popped_event[0]] = popped_event[1]

	return misaligned_activities
	
def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in):
	
	normalized_dataset = dataset.copy()
	normalization_parameters = {}
	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				if np.any(column_values) == True:
					column_values_mean = np.mean(column_values)
					column_values_std = np.std(column_values)
					if column_values_std != 0:
						column_values = (column_values - column_values_mean)/column_values_std
				else:
					column_values_mean = 0
					column_values_std = 0
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				if std != 0:
					parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
	
	return normalized_dataset, normalization_parameters		
	
def write_data(training_set, validation_set, test_set):

	training_set.to_csv(output_data_dir + "Training.csv", index = False)
	validation_set.to_csv(output_data_dir + "Validation.csv", index = False)
	test_set.to_csv(output_data_dir + "Test.csv", index = False)

	return None
	
try:
	fe_type = sys.argv[1]
except IndexError:
	print("Enter the right number of input arguments.")
	sys.exit()
	
normative_model = read_normative_model()
training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs = read_event_logs()
training_set, validation_set, test_set = feature_extraction(training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs, normative_model, fe_type)
write_data(training_set, validation_set, test_set)