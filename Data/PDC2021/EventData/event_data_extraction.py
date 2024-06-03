import os
import sys
import pandas as pd
import math
import random

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pm4py


input_dir = "Input/"
input_normal_dir = input_dir + "Normal/"
input_anomalous_dir = input_dir + "Anomalous/"

output_dir = "Output/"

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

def read_event_logs():

	normal_event_log = None
	anomalous_event_log = None

	normal_traces = []
	for event_log_filename in os.listdir(input_normal_dir):
		event_log = xes_importer.apply(input_normal_dir + event_log_filename)
		for trace in event_log:
			events = []
			for event in trace:
				events.append(event["concept:name"])
			try:
				if trace.attributes["pdc:isPos"] == True:
					normal_traces.append(events)
					temp_idx = temp_idx + 1
			except:
				normal_traces.append(events)
	
	
	#n_normal_test_traces = int(len(normal_traces)*0.25)
	#normal_test_traces = [normal_traces.pop(random.randrange(len(normal_traces))) for _ in range(n_normal_test_traces)]
	normal_event_log, _ = build_event_logs(normal_traces, [])

	anomalous_traces = []
	for event_log_filename in os.listdir(input_anomalous_dir):
		event_log = xes_importer.apply(input_anomalous_dir + event_log_filename)
		for trace in event_log:
			events = []
			for event in trace:
				events.append(event["concept:name"])
			try:
				if trace.attributes["pdc:isPos"] == True:
					anomalous_traces.append(events)
			except:
				anomalous_traces.append(events)
	
	_, anomalous_event_log = build_event_logs([], anomalous_traces)
	
	
	return normal_event_log, anomalous_event_log
	
def build_event_logs(normal_traces, anomalous_traces):
	normal_event_log = []
	anomalous_event_log = []
	
	for trace_idx, normal_trace in enumerate(normal_traces):
		caseid = trace_idx
		for event_idx, event in enumerate(normal_trace):
			event_timestamp = timestamp_builder(caseid)
			event = [caseid, event, event_timestamp]
			normal_event_log.append(event)
	
	normal_event_log = pd.DataFrame(normal_event_log, columns=['CaseID', 'Event', 'Timestamp'])
	normal_event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	normal_event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	normal_event_log = dataframe_utils.convert_timestamp_columns_in_df(normal_event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	normal_event_log = log_converter.apply(normal_event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	
	for trace_idx, anomalous_trace in enumerate(anomalous_traces):
		caseid = trace_idx
		for event_idx, event in enumerate(anomalous_trace):
			event_timestamp = timestamp_builder(caseid)
			event = [caseid, event, event_timestamp]
			anomalous_event_log.append(event)
	
	anomalous_event_log = pd.DataFrame(anomalous_event_log, columns=['CaseID', 'Event', 'Timestamp'])
	anomalous_event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	anomalous_event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	anomalous_event_log = dataframe_utils.convert_timestamp_columns_in_df(anomalous_event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	anomalous_event_log = log_converter.apply(anomalous_event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	
	return normal_event_log, anomalous_event_log
	
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
	
def write_event_logs(normal_event_log, anomalous_event_log):

	xes_exporter.apply(normal_event_log, output_dir + "Normal.xes")
	xes_exporter.apply(anomalous_event_log, output_dir + "Anomalous.xes")

	return None
	
normal_event_log, anomalous_event_log = read_event_logs()
write_event_logs(normal_event_log, anomalous_event_log)





