import pandas as pd 
import wandb
import json
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("patchouli_lab/leave_one_out")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("leave_one_out.csv")

def preprocess_json(json_str):
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", "\"")
    return json.loads(json_str)

# Read the exported CSV file
df = pd.read_csv('leave_one_out.csv')

# Expand the 'summary' and 'config' columns from JSON strings to separate columns
summary_expanded = df['summary'].apply(preprocess_json).apply(pd.Series)
config_expanded = df['config'].apply(preprocess_json).apply(pd.Series)

# Combine the expanded columns with the original dataframe
df_expanded = pd.concat([df.drop(columns=['summary', 'config']), summary_expanded, config_expanded], axis=1)

# Export the processed dataframe to a new CSV file
df_expanded.to_csv('leave_one_out_processed.csv', index=False)

print("Data processed and exported to leave_one_out_processed.csv")

