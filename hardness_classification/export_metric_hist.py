import wandb
import pandas as pd
import csv
import argparse
# Authenticate your WandB account
wandb.login()

# Initialize API
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("patchouli_lab/leave_one_out")

def get_wandb_field_history(username, project_name, run_id, field_name):
    """
    Retrieves the history of a specific field (metric or parameter) from a Wandb run using the Wandb API.
    
    Parameters:
    - username (str): Your Wandb username.
    - project_name (str): Name of the project containing the run.
    - run_id (str): ID of the specific run.
    - field_name (str): Name of the field (metric or parameter) to retrieve.
    
    Returns:
    - list: A list of tuples (step, value) representing the history of the field.
    """
    # Initialize the Wandb API
    api = wandb.Api()
    
    # Fetch the run
    run = api.run(f"{username}/{project_name}/{run_id}")
    
    # Access the history of the specified field
    history = run.history(keys=[field_name])
    history_list = [run.name] + ([(row[field_name]) for _, row in history.iterrows()])
    
    return history_list




# Example usage:
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Export metric history from WandB runs to CSV files.")
    parser.add_argument("--username", type=str, default="patchouli_lab", help="Your Wandb username.")
    parser.add_argument("--project_name", type=str, default="Final_exp_Aug_12_yuan_DINOKeras", help="Name of the project containing the run.")
    parser.add_argument("--field_name", type=str, default="acc_comp", help="Name of the field (metric or parameter) to retrieve.")
    args = parser.parse_args()
    username = args.username
    field_name = args.field_name
    # project_name = "Comp_Model_Aug_11_hardness_ConvLSTM"
    # project_name = "Comp_Model_Aug_11_hardness_DINOv2NN"
    project_name = args.project_name

    runs = api.runs(f"{username}/{project_name}")
    # Replace with your Wandb username, project name, run ID, and field name
    csv_filename = f"metrics/exp/{project_name}_{field_name}_history.csv"
    
    with open(csv_filename, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile)      

        for i in range(0, len(runs)):
            run_id = runs[i].id                
            # Retrieve the history of the specified field
            history_list = get_wandb_field_history(username, project_name, run_id, field_name)
            csvwriter.writerow(history_list)

    print(f"History of '{field_name}' written to '{csv_filename}'.")
