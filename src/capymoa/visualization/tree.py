from __future__ import annotations

import numpy as np
from java.lang import System  # type: ignore

from capymoa.stream import Schema


def dot_escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', r"\"")


def _class_name(schema: Schema | None, index: int) -> str:
    if schema is None:
        return str(index)
    try:
        return str(schema.get_value_for_index(index))
    except Exception:
        return str(index)


def _existing_children(node):
    children = []
    for branch_idx in range(int(node.numChildren())):
        child = node.getChild(branch_idx)
        if child is not None:
            children.append((branch_idx, child))
    return children


def _feature_name(schema: Schema | None, index: int) -> str:
    if schema is None:
        return f"x[{index}]"

    try:
        return str(schema.get_moa_header().inputAttribute(index).name())
    except Exception:
        return f"x[{index}]"


def _format_instance_text(model, sample_instance) -> str:
    if sample_instance is None:
        return "Incoming instance: none"

    values = np.asarray(sample_instance.x, dtype=float)
    parts = []
    for idx, value in enumerate(values):
        name = _feature_name(getattr(model, "schema", None), idx)
        if np.isnan(value):
            parts.append(f"{name} = ?")
        else:
            parts.append(f"{name} = {value:.3g}")

    return "Incoming instance: " + ", ".join(parts)


def _format_prediction_text(model, sample_instance, proba=None) -> str:
    if sample_instance is None:
        return "Prediction: none"

    if proba is None:
        proba = model.predict_proba(sample_instance)
    if proba is None:
        return "Prediction: none"

    proba = np.asarray(proba, dtype=np.float64)
    pred_idx = int(np.argmax(proba))
    pred_label = _class_name(model.schema, pred_idx)
    proba_text = ", ".join(
        f"{_class_name(model.schema, idx)}: {value:.3f}"
        for idx, value in enumerate(proba)
    )
    return f"Prediction: {pred_label} | Proba: {proba_text}"


def _branch_label(model, split_test, branch_idx: int) -> str:
    try:
        cond = split_test.describeConditionForBranch(
            branch_idx, model.moa_learner.getModelContext()
        )
        cond = str(cond)
        if cond:
            return cond
    except Exception:
        pass
    return f"branch {branch_idx}"


def _split_node_label(model, node) -> str:
    split_test = node.getSplitTest()

    try:
        split_str = str(split_test)
        if split_str:
            return f"Split\n{split_str}"
    except Exception:
        pass

    branch_texts = []
    try:
        for branch_idx in range(int(node.numChildren())):
            cond = split_test.describeConditionForBranch(
                branch_idx, model.moa_learner.getModelContext()
            )
            branch_texts.append(str(cond))
    except Exception:
        pass

    if branch_texts:
        return "Split\n" + "\n".join(branch_texts)

    return "Split"


def _format_votes_text(model, votes, prefix: str) -> str:
    votes = np.asarray(votes, dtype=np.float64)
    if not model._has_usable_votes(votes):
        return f"{prefix}: none\npredict: none"

    pred_idx = int(np.argmax(votes))
    pred_name = _class_name(model.schema, pred_idx)
    parts = [
        f"{_class_name(model.schema, idx)}: {value:.2f}"
        for idx, value in enumerate(votes)
    ]
    return f"{prefix}: {', '.join(parts)}\npredict: {pred_name}"


def _best_effort_leaf_votes_without_instance(node) -> np.ndarray:
    try:
        dist = node.observedClassDistribution
        return np.asarray(list(dist.getArrayRef()), dtype=np.float64)
    except Exception:
        pass

    try:
        dist = node.getObservedClassDistribution()
        return np.asarray(list(dist.getArrayRef()), dtype=np.float64)
    except Exception:
        pass

    return np.array([], dtype=np.float64)


def _leaf_dot_label(model, node, java_instance, include_votes: bool) -> str:
    if java_instance is None:
        votes = _best_effort_leaf_votes_without_instance(node)
    else:
        votes = model._node_votes(node, java_instance)

    if not include_votes:
        votes = np.asarray(votes, dtype=np.float64)
        if not model._has_usable_votes(votes):
            return "Leaf\npredict: none"
        pred_idx = int(np.argmax(votes))
        pred_name = _class_name(model.schema, pred_idx)
        return f"Leaf\npredict: {pred_name}"

    return "Leaf\n" + _format_votes_text(model, votes, "votes")


def _node_key(node) -> int:
    return int(System.identityHashCode(node))


def _trace_node_keys(trace) -> set[int]:
    return {_node_key(node) for node in trace.nodes}


def _trace_edge_keys(trace) -> set[tuple[int, int, int]]:
    return {
        (_node_key(parent), branch_idx, _node_key(child))
        for parent, branch_idx, child in trace.edges
    }


def _trace_vote_source_node_key(trace) -> int | None:
    if trace.vote_source_node is None:
        return None
    return _node_key(trace.vote_source_node)


def instance_triggers_missing_value_path(model, sample_instance, node=None) -> bool:
    if sample_instance is None:
        return False

    if node is None:
        node = model.get_tree_root()

    if node is None or node.isLeaf():
        return False

    return _subtree_triggers_missing_value_path(
        java_instance=sample_instance.java_instance.getData(),
        node=node,
    )


def _subtree_triggers_missing_value_path(java_instance, node) -> bool:
    if node is None or node.isLeaf():
        return False

    split_test = node.getSplitTest()
    try:
        if not bool(split_test.resultKnownForInstance(java_instance)):
            return True
    except Exception:
        return False

    try:
        child = node.getChild(int(split_test.branchForInstance(java_instance)))
    except Exception:
        return False

    if child is None:
        return False

    return _subtree_triggers_missing_value_path(
        java_instance=java_instance,
        node=child,
    )


def export_hoeffding_tree_to_dot(
    model,
    sample_instance=None,
    title: str = "Hoeffding Tree",
    include_leaf_votes: bool = True,
    highlight_path: bool = False,
    require_missing_path: bool = False,
) -> str:
    root = model.get_tree_root()
    if root is None:
        return 'digraph HoeffdingTree { empty [label="Tree is empty"]; }'

    java_instance = None
    if sample_instance is not None:
        java_instance = sample_instance.java_instance.getData()

    if require_missing_path and not instance_triggers_missing_value_path(
        model, sample_instance, root
    ):
        raise ValueError(
            "sample_instance does not trigger missing-value handling on the tree path."
        )

    active_nodes = set()
    active_edges = set()
    trace_proba = None
    trace_vote_source_node_key = None
    trace_votes = None
    if highlight_path and java_instance is not None:
        trace = model.trace_prediction_path(java_instance, root)
        active_nodes = _trace_node_keys(trace)
        active_edges = _trace_edge_keys(trace)
        trace_vote_source_node_key = _trace_vote_source_node_key(trace)
        trace_votes = trace.votes
        if model.missing_value_policy != "default":
            trace_proba = model._normalize_votes(trace.votes)

    instance_text = _format_instance_text(model, sample_instance)
    prediction_text = _format_prediction_text(model, sample_instance, proba=trace_proba)

    graph_label = f"{title}\n{instance_text}\n{prediction_text}"

    lines = [
        "digraph HoeffdingTree {",
        f'  graph [rankdir=TB, labelloc=t, fontsize=18, label="{dot_escape(graph_label)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    visited = {}
    counter = {"value": 0}

    def next_id():
        node_id = f"n{counter['value']}"
        counter["value"] += 1
        return node_id

    def style_node(node_obj, is_leaf: bool) -> str:
        if _node_key(node_obj) in active_nodes:
            return 'fillcolor="#dbeafe", color="#1d4ed8", penwidth=2'
        if is_leaf:
            return 'fillcolor="#f0fdf4", color="#16a34a"'
        return 'fillcolor="#f8fafc", color="#94a3b8"'

    def style_edge(edge_key) -> str:
        if edge_key in active_edges:
            return 'color="#1d4ed8", penwidth=2'
        return 'color="#94a3b8", style="dashed"'

    def walk(node):
        node_key = _node_key(node)
        if node_key in visited:
            return visited[node_key]

        node_id = next_id()
        visited[node_key] = node_id

        is_leaf = bool(node.isLeaf())

        if is_leaf:
            label = _leaf_dot_label(
                model=model,
                node=node,
                java_instance=java_instance,
                include_votes=include_leaf_votes,
            )
            lines.append(
                f'  {node_id} [label="{dot_escape(label)}", {style_node(node, True)}];'
            )
            return node_id

        split_label = _split_node_label(model, node)
        if (
            include_leaf_votes
            and trace_vote_source_node_key is not None
            and trace_vote_source_node_key == _node_key(node)
            and trace_votes is not None
        ):
            split_label += "\n" + _format_votes_text(
                model, trace_votes, "fallback votes"
            )
        lines.append(
            f'  {node_id} [label="{dot_escape(split_label)}", {style_node(node, False)}];'
        )

        split_test = node.getSplitTest()

        for branch_idx, child in _existing_children(node):
            child_id = walk(child)
            edge_label = _branch_label(model, split_test, branch_idx)
            edge_key = (_node_key(node), branch_idx, _node_key(child))

            lines.append(
                f'  {node_id} -> {child_id} [label="{dot_escape(edge_label)}", {style_edge(edge_key)}];'
            )

        return node_id

    walk(root)
    lines.append("}")
    return "\n".join(lines)
