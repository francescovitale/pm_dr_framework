from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
import pm4py

import os
import sys
import math
import random


input_dir = "Input/PP/"
input_eventdata_dir = input_dir + "EventData/"

output_dir = "Output/PP/"
output_eventlogs_dir = output_dir + "EventLogs/"
output_test_eventlogs_dir = output_eventlogs_dir + "Test/"
output_training_eventlogs_dir = output_eventlogs_dir + "Training/"
output_validation_eventlogs_dir = output_eventlogs_dir + "Validation/"


def read_event_logs():

	event_logs = {}

	for filename in os.listdir(input_eventdata_dir):
		event_logs[filename.split(".xes")[0]] = xes_importer.apply(input_eventdata_dir + filename)

	return event_logs
	
def split_event_logs(event_logs, n_traces_per_log, validation_percentage, test_percentage):
	training_event_logs = []
	validation_event_logs = []
	anomalous_test_event_logs = []
	normal_test_event_logs = []
	
	# Training, validation and normal test event logs
	normal_event_log = event_logs["Normal"]
	n_normal_event_logs = math.floor(len(normal_event_log)/n_traces_per_log)
	n_normal_test_event_logs = math.floor(n_normal_event_logs*test_percentage)
	n_validation_event_logs = math.floor((n_normal_event_logs-n_normal_test_event_logs)*validation_percentage)
	normal_event_logs = []
	for i in range(0,n_normal_event_logs):
		normal_event_logs.append(normal_event_log[i*n_traces_per_log:i*n_traces_per_log+n_traces_per_log])
	normal_test_event_logs = [normal_event_logs.pop(random.randrange(len(normal_event_logs))) for _ in range(n_normal_test_event_logs)]
	validation_event_logs = [normal_event_logs.pop(random.randrange(len(normal_event_logs))) for _ in range(n_validation_event_logs)]
	training_event_logs = normal_event_logs

	# Anomalous test event logs
	anomalous_test_event_log = event_logs["Anomalous"]
	n_anomalous_event_logs = math.floor(len(anomalous_test_event_log)/n_traces_per_log)
	for i in range(0,n_anomalous_event_logs):
		anomalous_test_event_logs.append(anomalous_test_event_log[i*n_traces_per_log:i*n_traces_per_log+n_traces_per_log])
	
	return training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs

def write_event_logs(training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs):

	for idx, event_log in enumerate(training_event_logs):
		pm4py.write_xes((pm4py.objects.log.obj.EventLog)(event_log), output_training_eventlogs_dir + "TR_" + str(idx) + ".xes")
		
	for idx, event_log in enumerate(validation_event_logs):
		pm4py.write_xes((pm4py.objects.log.obj.EventLog)(event_log), output_validation_eventlogs_dir + "VAL_" + str(idx) + ".xes")	
		
	for idx, event_log in enumerate(normal_test_event_logs):
		pm4py.write_xes((pm4py.objects.log.obj.EventLog)(event_log), output_test_eventlogs_dir + "TST_N_" + str(idx) + ".xes")
		
	for idx, event_log in enumerate(anomalous_test_event_logs):
		pm4py.write_xes((pm4py.objects.log.obj.EventLog)(event_log), output_test_eventlogs_dir + "TST_A_" + str(idx) + ".xes")	
	
	return None
	
	
try:
	n_traces_per_log = int(sys.argv[1])
	validation_percentage = float(sys.argv[2])
	test_percentage = float(sys.argv[3])

except IndexError:
	print("Enter the right number of input arguments")
	sys.exit()

event_logs = read_event_logs()
training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs = split_event_logs(event_logs, n_traces_per_log, validation_percentage, test_percentage)
write_event_logs(training_event_logs, validation_event_logs, normal_test_event_logs, anomalous_test_event_logs)

