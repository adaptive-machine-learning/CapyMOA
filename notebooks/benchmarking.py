# Python imports
import csv
import pandas as pd
from river.forest import ARFClassifier
from river.ensemble import SRPClassifier
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from river.neighbors import KNNClassifier, LazySearch
from river import metrics
from river import stream

# Library imports
from capymoa.evaluation.evaluation import *
from capymoa.stream import stream_from_file

# MOA/Java imports
from moa.classifiers.meta import AdaptiveRandomForest, StreamingRandomPatches
from moa.classifiers.trees import HoeffdingTree, EFDT
from moa.classifiers.lazy import kNN
from moa.classifiers.bayes import NaiveBayes
from capymoa.base import MOAClassifier

MAX_INSTANCES = 100
OUTPUT_FILE_PATH = "./experiments/experiments_MOA_ARF_2.csv"
OUTPUT_FILE_RIVER_PATH = "./experiments/experiments_RIVER_2.csv"

## Datasets paths
arff_RTG_2abrupt_path = "./data/RTG_2abrupt.arff"
csv_RTG_2abrupt_path = "./data/RTG_2abrupt.csv"


## Function to abstract the test and train loop using RIVER
def run_test_then_train_RIVER(dataset, model, max_instances=1000, sample_frequency=100):
    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()

    instancesProcessed = 1
    accuracy = metrics.Accuracy()

    X, Y = dataset[:, :-1], dataset[:, -1]

    data = []
    performance_names = ["Classified instances", "accuracy"]
    performance_values = []

    ds = stream.iter_array(X, Y)

    for x, y in ds:
        if instancesProcessed > max_instances:
            break
        yp = model.predict_one(x)
        accuracy.update(y, yp)
        model.learn_one(x, y)

        if instancesProcessed % sample_frequency == 0:
            performance_values = [instancesProcessed, accuracy.get()]
            data.append(performance_values)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    return (
        accuracy.get(),
        elapsed_wallclock_time,
        elapsed_cpu_time,
        pd.DataFrame(data, columns=performance_names),
    )


def run_MOA_experiment(
    arff_path, model, CLI="", output_file_path=OUTPUT_FILE_PATH
):
    with open(output_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)


        data_arff = stream_from_file(path_to_csv_or_arff=arff_path)
        model = MOAClassifier(moa_learner=model(), CLI=CLI, schema=data_arff.get_schema())

        results = test_then_train_evaluation(
            stream=data_arff,
            learner=model,
            max_instances=MAX_INSTANCES,
            sample_frequency=MAX_INSTANCES,
        )
        print(
            f"{arff_path}, {model.__str__()} {CLI}, {results['cumulative'].accuracy():.4f}, {results['wallclock']:.4f}, {results['cpu_time']:.4f}"
        )
        writer.writerow(
            [arff_path, model.__str__() + CLI, results['cumulative'].accuracy(), results['wallclock'], results['cpu_time']]
        )


def run_RIVER_experiment(
    csv_path, model=ARFClassifier(), CLI="", output_file_path=OUTPUT_FILE_RIVER_PATH
):
    with open(output_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        data_csv = pd.read_csv(csv_path).to_numpy()

        acc, wallclock, cpu_time, df = run_test_then_train_RIVER(
            dataset=data_csv,
            model=model,
            max_instances=MAX_INSTANCES,
            sample_frequency=MAX_INSTANCES,
        )
        print(
            f"{csv_path}, {model.__class__.__name__} {CLI}, {acc:.4f}, {wallclock:.4f}, {cpu_time:.4f}"
        )
        writer.writerow(
            [csv_path, model.__class__.__name__ + CLI, acc, wallclock, cpu_time]
        )


def experiments_MOA():
    with open(OUTPUT_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write the header row if the file is empty
        if file.tell() == 0:
            writer.writerow(
                ["dataset", "classifier", "accuracy", "wallclock(s)", "cpu_time(s)"]
            )

    run_MOA_experiment(arff_path=arff_RTG_2abrupt_path, model=NaiveBayes, CLI="")
    run_MOA_experiment(arff_path=arff_RTG_2abrupt_path, model=HoeffdingTree, CLI="")
    run_MOA_experiment(arff_path=arff_RTG_2abrupt_path, model=EFDT, CLI="")
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path, model=kNN, CLI=" -w 1000 -k 3"
    )

    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=AdaptiveRandomForest,
        CLI="-s 5 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=AdaptiveRandomForest,
        CLI="-s 10 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=AdaptiveRandomForest,
        CLI="-s 30 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=AdaptiveRandomForest,
        CLI="-s 100 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=AdaptiveRandomForest,
        CLI="-s 100 -j 4 -o (Percentage (M * (m / 100))) -m 60",
    )

    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=StreamingRandomPatches,
        CLI="-s 5 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=StreamingRandomPatches,
        CLI="-s 10 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=StreamingRandomPatches,
        CLI="-s 30 -o (Percentage (M * (m / 100))) -m 60",
    )
    run_MOA_experiment(
        arff_path=arff_RTG_2abrupt_path,
        model=StreamingRandomPatches,
        CLI="-s 100 -o (Percentage (M * (m / 100))) -m 60",
    )


def experiments_RIVER():
    with open(OUTPUT_FILE_RIVER_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write the header row if the file is empty
        if file.tell() == 0:
            writer.writerow(
                ["dataset", "classifier", "accuracy", "wallclock(s)", "cpu_time(s)"]
            )

    run_RIVER_experiment(csv_path=csv_RTG_2abrupt_path, model=GaussianNB(), CLI="")
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path, model=HoeffdingTreeClassifier(), CLI=""
    )
    # Error while executing!
    # run_RIVER_experiment(csv_path=csv_RTG_2abrupt_path, model=ExtremelyFastDecisionTreeClassifier(), CLI="")
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=KNNClassifier(engine=LazySearch(window_size=1000), n_neighbors=3),
        CLI="engine=LazySearch(window_size=1000), n_neighbors=3",
    )

    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=ARFClassifier(n_models=5, max_features=0.60),
        CLI="n_models=5,max_features=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=ARFClassifier(n_models=10, max_features=0.60),
        CLI="n_models=10,max_features=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=ARFClassifier(n_models=30, max_features=0.60),
        CLI="n_models=30,max_features=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=ARFClassifier(n_models=100, max_features=0.60),
        CLI="n_models=100,max_features=0.60",
    )

    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=SRPClassifier(n_models=5, subspace_size=0.60),
        CLI="n_models=5,subspace_size=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=SRPClassifier(n_models=10, subspace_size=0.60),
        CLI="n_models=10,subspace_size=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=SRPClassifier(n_models=30, subspace_size=0.60),
        CLI="n_models=30,subspace_size=0.60",
    )
    run_RIVER_experiment(
        csv_path=csv_RTG_2abrupt_path,
        model=SRPClassifier(n_models=100, subspace_size=0.60),
        CLI="n_models=100,subspace_size=0.60",
    )


if __name__ == "__main__":
    experiments_MOA()
    experiments_RIVER()
