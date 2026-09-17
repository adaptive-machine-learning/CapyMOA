# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Save and Load a Model
#
# In this tutorial, we illustrate the process of saving and loading a model using CapyMOA. 
#
# * We use the SEA synthetic generator as the data source, and the AdaptiveRandomForestClassifier as the learner.
# * The trained model is saved to a file, specifically `capymoa_ARF_model.pkl`.
# * Subsequently, we reload the model from the file and resume training and evaluating its performance on the SEA data.
# * As a final step, we delete the model file.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 01/12/2025**

# %% [markdown]
# ## Training and saving the model
#
# * We train the model on 5k instances from SEA using the `evaluate_prequential` function.
# * We proceed to save the model with `save_model(learner, "capymoa_ARF_model.pkl")`.

# %%
from capymoa.classifier import AdaptiveRandomForestClassifier
from capymoa.core.io import load_model, save_model
from capymoa.evaluation import prequential_evaluation
from capymoa.stream.generator import SEA

stream = SEA()
learner = AdaptiveRandomForestClassifier(schema=stream.get_schema(), ensemble_size=10)

results = prequential_evaluation(stream=stream, learner=learner, max_instances=5000)

print(f"Accuracy: {results['cumulative'].accuracy():.2f}")

with open("capymoa_ARF_model.pkl", "wb") as f:
    save_model(learner, f)

# %% [markdown]
# ## Loading and resuming training
#
# * We use `os.path.getsize()` to inspect the size (KB) of the saved file.
# * We don't restart the synthetic stream, we just continue processing it through another call to `prequential_evaluation`.
# * Finally, we observe the accuracy.

# %%
import os

model_file = "capymoa_ARF_model.pkl"

model_size = os.path.getsize(model_file)
print(f"The saved model size: {model_size / 1024:.2f} KB")

with open(model_file, "rb") as f:
    restored_learner = load_model(f)

# Train for more 50k instances on the restored model
results = prequential_evaluation(
    stream=stream, learner=restored_learner, max_instances=5000
)

print(f"Updated accuracy: {results['cumulative'].accuracy():.2f}")

# %% [markdown]
# ## Cleanup 
#
# * As a last step, we delete the model.

# %%
if os.path.exists(model_file):
    os.remove(model_file)
    print(f"File {model_file} has been deleted.")
else:
    print(f"File {model_file} not found.")
