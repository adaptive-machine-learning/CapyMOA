#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

from openmoa.classifier import FESLClassifier
from openmoa.evaluation import prequential_evaluation
from openmoa.stream import ARFFStream

FILE_PATH = Path(r"C:/reposity clone/OpenMOA/datasets/FESL/australian_perm5.arff")


def main(file_path: str | Path = FILE_PATH) -> None:
    stream = ARFFStream(str(file_path))

    n_smp = 690
    max_instances = n_smp

    fesl = FESLClassifier(
        schema=stream.schema,
        s1_feature_indices=list(range(42)),
        s2_feature_indices=list(range(42, 71)),
        overlap_size=10,
        switch_point=345,
        ensemble_method="selection",
        learning_rate_scale=1.0,
        random_seed=None,
    )

    results = prequential_evaluation(
        stream,
        fesl,
        max_instances=max_instances,
        window_size=1,
        progress_bar=True,
    )
    print(f"Accuracy: {results['cumulative'].accuracy():.3f}%")


if __name__ == "__main__":
    main()