import time
import mteb
import importlib.metadata as metadata
from importlib.metadata import version
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "all-MiniLM-L6-v2"

start_time = time.time()  # Record the start time

model = SentenceTransformer("all-MiniLM-L6-v2")
evaluation = MTEB(tasks=["ClimateFEVER"])
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results/{model_name}")

end_time = time.time()  # Record the end time

# Calculate the duration
duration_seconds = end_time - start_time
hours = int(duration_seconds // 3600)
minutes = int((duration_seconds % 3600) // 60)
seconds = int(duration_seconds % 60)

print(f"Model evaluation took: {hours} hours, {minutes} minutes, and {seconds} seconds.")
