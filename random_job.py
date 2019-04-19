###### Instructions
# This function works with googlecluster traces.
# Example of usage within your python script:
#       extracted_job = random_job()
# The result is a dataframe with the job characteristics.
#
# Before using it, you need to perform the following steps:
# 1. Download the data in your favorite location <loc> in your local machine.
# 2. Write this location in the dataset_path variable
# 3. Recreate in your local machine the folders <loc>/job_events/ and <loc>/task_events.
# 4. Put there the job_events and task_events file. Note that, if you have limitations
# in downloading or of space in your local machine,  you don't need to retrieve
# all the files from googlecluster traces. You can just retrieve some parts. 
# Be sure that you download exactly the same parts of task_events and job_events.



import glob
import pandas as pd
import numpy as np
import math
import numpy.random as Random

def random_job_attempt():
    #### INPUT PARAMETERS
    DEBUGG = False
    #dataset_path = "/home/araldo/tsp/formation/machine_learning/datasets/googlecluster/"
    dataset_path = "datasets/"
    # The following information is taken from schema.csv provided by google
    job_id_field_in_job_events = 2 # This is to avoid the overhead of loading the headers
    event_type_field_in_job_events = 3
    event_type = 0 # SUBMIT
    job_id_field_in_task_events = 2
    event_type_field_in_task_events = 5
    CPU_field = 9
    memory_field = 10
    disk_field = 11
    task_id_field = 3
    
    
    #### Select a random job
    ## Select a random file
    # source: https://stackoverflow.com/a/41447012/2110769
    job_filename_string = dataset_path+"job_events/part-?????-of-?????.csv.gz"
    job_files = [f for f in glob.glob(job_filename_string)]
    if DEBUGG:
        print("job_files are",job_files)
    r = math.ceil(Random.uniform(0, len(job_files)-1))
    job_file = job_files[r]
    if DEBUGG:
        print("file selected: ", job_file)
        
    job_df = pd.read_csv(job_file, header=None)

    # Select only the event type that we want
    job_df = job_df[ job_df[event_type_field_in_job_events] == event_type ]
    
    job_ids = job_df[job_id_field_in_job_events]
    job_ids = np.unique(job_ids)
    if DEBUGG:
        num_of_unique_job_ids = len(job_ids)
        print("There are ",num_of_unique_job_ids," unique job_ids")
         
    # Select a random job_id
    r = math.ceil(Random.uniform(0, len(job_ids)-1))
    job_id = job_ids[r]
    
    if DEBUGG:
        print("job_id=",job_id," has been selected")
        
    # Open the corresponding task_events file
    splitted = job_file.split("/")
    name_nucleous = splitted[len(splitted)-1]
    
    task_file = dataset_path+"task_events/"+name_nucleous
    if DEBUGG:
        print("Opening ",task_file)
        
    task_df = pd.read_csv(task_file, header=None)
    
    
    # Select only the tasks that are related to the extracted job and only the submission events
    task_df_selected = task_df[ \
        task_df[job_id_field_in_task_events] == job_id ]
    task_df_selected = task_df_selected[ \
        task_df_selected[event_type_field_in_task_events] ==  event_type] 
    task_df_selected = task_df_selected[[job_id_field_in_task_events,task_id_field,CPU_field,memory_field,disk_field]]
    task_df_selected.columns = ["job_id","task_id","CPU","memory","disk"]
    
    
    return task_df_selected


def random_job():
    job_found = False
    job_description = []
    while (not job_found):
        job_description = random_job_attempt()
        if job_description.shape[0] > 0:
            job_found = True
    return job_description

if __name__ == "__main__":
    print(random_job())