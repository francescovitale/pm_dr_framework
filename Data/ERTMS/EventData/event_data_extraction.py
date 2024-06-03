from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pm4py.algo.discovery as pdiscovery
import pm4py
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from scipy.stats import norm
from scipy.stats import uniform

input_dir = "Input/"

output_dir = "Output/"


def get_normative_model():
	normative_model = pm4py.read_bpmn(input_dir + "RBC_HANDOVER.bpmn")
	net, im, fm = pm4py.convert_to_petri_net(normative_model)
	
	normative_model = {}
	normative_model["network"] = net
	normative_model["initial_marking"] = im
	normative_model["final_marking"] = fm
	activities = {}
	activities_file = open(input_dir + "activities", "r")
	lines = activities_file.readlines()
	for line in lines:
		line = line.replace("\n","")
		tokens = line.split(":")[1]
		activities[tokens.split(",")[0]] = tokens.split(",")[1]
	activities_file.close()
	
	return normative_model, activities

def playout_model(normative_model, n_traces_per_log, n_logs_per_type):

	total_n_traces = n_traces_per_log * 2 * n_logs_per_type
	event_log = simulator.apply(normative_model["network"], normative_model["initial_marking"], variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: total_n_traces})
	return event_log
	
def split_event_log(event_log, n_traces_per_event_log, activities):
	event_logs = []
	n_event_logs = math.ceil(len(event_log)/n_traces_per_event_log)
	
	for i in range(0, n_event_logs):
		if i < n_event_logs-1:
			traces = event_log[i*n_traces_per_event_log:i*n_traces_per_event_log+n_traces_per_event_log]
			new_traces = []
			temp = {}
			temp["activities"] = []
			temp["resources"] = []
			for trace in traces:
				trace_activities = []
				trace_resources = []
				for event in trace:
					trace_activities.append(event["concept:name"])
					trace_resources.append(activities[event["concept:name"]])
				temp["activities"] = trace_activities
				temp["resources"] = trace_resources
				new_traces.append(temp)
			event_logs.append(new_traces)
		else:
			traces = event_log[i*n_traces_per_event_log:i*n_traces_per_event_log+n_traces_per_event_log]
			new_traces = []
			temp = {}
			temp["activities"] = []
			temp["resources"] = []
			for trace in traces:
				trace_activities = []
				trace_resources = []
				for event in trace:
					trace_activities.append(event["concept:name"])
					trace_resources.append(activities[event["concept:name"]])
				temp["activities"] = trace_activities
				temp["resources"] = trace_resources
				new_traces.append(temp)
			event_logs.append(new_traces)
		
	return event_logs	
	
def get_per_event_log_traces(event_logs):

	per_event_log_traces = {}

	for event_log in event_logs:
		traces = []
		for trace in event_logs[event_log]:
			events = []
			for event in trace:
				events.append(event["concept:name"])
				
			traces.append(events)
		per_event_log_traces[event_log] = traces		
			
	
	return per_event_log_traces	
	
def inject_anomaly(injection_probability, probability_distribution):
	inject_anomaly = False
	random_number = 0.0

	if probability_distribution == "uniform":
		random_number = np.random.uniform()
		if uniform.cdf(random_number) > 1-injection_probability:
			inject_anomaly = True

	elif probability_distribution == "normal":
		random_number = np.random.norm()
		if norm.cdf(random_number) > 1-injection_probability:
			inject_anomaly = True

	return inject_anomaly	
	
def inject_resource_anomalies(trace, probability_distribution, injection_probability, faulty_resource):
	
	# missed activities
	
	trace_length = len(trace["activities"])

	for idx, resource in enumerate(trace["resources"]):
		if resource == faulty_resource:
			if inject_anomaly(injection_probability, probability_distribution) == True:
				del trace["activities"][idx]
				del trace["resources"][idx]

	# wrongly ordered activities
	
	trace_length = len(trace["activities"])
	temp_activity = ""
	temp_resource = ""
	
	for idx, resource in enumerate(trace["resources"]):
		if resource == faulty_resource:
			if inject_anomaly(injection_probability, probability_distribution) == True:
				
				idx_to_swap = random.choice(range(0, trace_length-1))
				temp_activity = trace["activities"][idx]
				temp_resource = trace["resources"][idx]

				trace["activities"][idx] = trace["activities"][idx_to_swap]
				trace["resources"][idx] = trace["resources"][idx_to_swap]

				trace["activities"][idx_to_swap] = temp_activity
				trace["resources"][idx_to_swap] = temp_resource

	# duplicated activities
	
	skip = False

	for idx, resource in enumerate(trace["resources"]):
		if resource == faulty_resource:
			if skip == False:
				skip = True
				if inject_anomaly(injection_probability, probability_distribution) == True:
					trace["activities"].insert(idx,trace["activities"][idx])
					trace["resources"].insert(idx,trace["resources"][idx])
			else:
				skip = False
			
	return trace	
	
def inject_trace(trace, anomalous_resource, probability_distribution, injection_probability, percentages, resources_list, event_log_type):

	noisy_trace = inject_resource_anomalies(trace, "uniform", injection_probability, anomalous_resource)

	return noisy_trace	
	
def inject_event_log(event_log, activities, resources, n_traces_per_log, n_logs_per_type):

	percentages = generate_injection_percentages(resources, "uniform", n_traces_per_log*n_logs_per_type)
	event_logs = split_event_log(event_log, n_traces_per_log, activities)
	normal_event_log = []
	for i in range(0,n_logs_per_type):
		normal_event_log = normal_event_log + event_logs[i]
	anomalous_event_log = []
	for i in range(n_logs_per_type,len(event_logs)):
		anomalous_event_log = anomalous_event_log + event_logs[i]
	
	temp = []
	for trace in normal_event_log:
		anomalous_resource = random.choice(resources)
		new_trace = inject_trace(trace.copy(), anomalous_resource, None, 0.05, None, resources.copy(), "Normal")
		temp.append(new_trace)
	normal_event_log = temp	
	
	temp = []
	for idx, trace in enumerate(anomalous_event_log):
		anomalous_resource = random.choice(resources)
		new_trace = inject_trace(trace.copy(), anomalous_resource, None, 0.25, None, resources.copy(), "Anomalous")
		temp.append(new_trace)
	anomalous_event_log = temp	

	return normal_event_log, anomalous_event_log
	
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

def write_event_logs(normal_event_log, anomalous_event_log):

	pm4py.write_xes((pm4py.objects.log.obj.EventLog)(normal_event_log), output_dir + "Normal.xes")
	pm4py.write_xes((pm4py.objects.log.obj.EventLog)(anomalous_event_log), output_dir + "Anomalous.xes")

	return None
	
def generate_injection_percentages(resources_list, probability_distribution, n_percentages):

	percentages = []
	for i in range(0, n_percentages):
		run_percentage = {}

		percentages_sum = 0.0
		random.shuffle(resources_list)

		low = 0.0
		high = 1.0
		for idx,resource in enumerate(resources_list):
			if(idx < len(resources_list)-1):
				#print("Percentage bounds for resource " + resource + "= [" + str(low) + "," + str(high) + "]")
				if probability_distribution == "uniform":
					percentage = np.random.uniform(low=low, high=high)
				elif probability_distribution == "normal":
					percentage = np.random.normal(loc=high/2, scale=high/6)
				elif probability_distribution == "lognormal":
					percentage = np.random.lognormal(mean=high/2, sigma=3/2)
					percentage = min(1.0-percentage, 1.0)
					percentage = max(0.0, percentage)
				run_percentage[resource] = percentage
				percentages_sum = percentages_sum + percentage
				#print("Percentage for resource " + resource + "=" + str(percentage))
				high = min(1.0 - percentages_sum, 1.0)
			else:
				percentage = 1.0 - percentages_sum
				percentage = max(0.0, percentage)
				run_percentage[resource] = percentage
				#print("Percentage for resource RTM = " + str(percentage))
				percentages_sum = percentages_sum + percentage
		percentages.append(run_percentage)		

	return percentages	
	
def write_normal_labels(normal_labels, cycle):

	file = open(output_firstrun_normallabels_dir + "NormalLabels_" + str(cycle) + ".txt", "w")
	
	for idx,label in enumerate(normal_labels):	
		if idx < len(normal_labels)-1:
			file.write(label + "\n")
		else:
			file.write(label)
	
	file.close()

	return None
	
def timestamp_builder(number):
	SSS = number
	ss = int(math.floor(SSS/1000))
	mm = int(math.floor(ss/60))
	hh = int(math.floor(mm/24))
	
	SSS = SSS % 1000
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)+"."+str(SSS)
	
def convert_event_log_to_xes(event_log):
	
	n = 0
	events = []
	for idx,trace in enumerate(event_log):
		for activity in trace["activities"]:
			events.append([idx,activity,timestamp_builder(n)])
			n = n+1
	new_event_log = pd.DataFrame(data=events, columns=['CaseID', 'Event', 'Timestamp'])
	new_event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	new_event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	new_event_log = dataframe_utils.convert_timestamp_columns_in_df(new_event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	new_event_log = log_converter.apply(new_event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	return new_event_log
	
	
try:
	n_traces_per_log = int(sys.argv[1])
	n_logs_per_type = int(sys.argv[2])

except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()

normal_event_log = []
anomalous_event_log = []
normative_model, activities = get_normative_model()
resources = list(set(list(activities.values())))
event_log = playout_model(normative_model, n_traces_per_log, n_logs_per_type)
normal_event_log, anomalous_event_log = inject_event_log(event_log, activities, resources, n_traces_per_log, n_logs_per_type)
normal_event_log = convert_event_log_to_xes(normal_event_log)
anomalous_event_log = convert_event_log_to_xes(anomalous_event_log)
write_event_logs(normal_event_log, anomalous_event_log)
