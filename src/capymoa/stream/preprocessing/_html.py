"""Render a pipeline as HTML for Jupyter.

A pipeline shows as a column of boxes, one per step, with nested pipelines drawn
as a box inside a box. Click a step to see its parameters. Each step also shows
the schema it works on, when that is known.

The look follows scikit-learn's estimator display, but this does not use
scikit-learn's renderer: that is driven by private hooks
(``_sk_visual_block_`` in ``sklearn.utils._repr_html``), and it calls
``get_params(deep=...)``, which CapyMOA's ``get_params`` does not accept.

Styles are scoped to one id per render, so two pipelines in a notebook do not
clash. Viewers that strip styles (some notebook previews) show the plain-text
form instead.
"""

from __future__ import annotations

import html
import itertools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import BasePipeline, PipelineElement

_render_ids = itertools.count()

#: Longest parameter value shown before it is cut short.
_MAX_VALUE_CHARS = 60

# `#ROOT` is replaced with the id of each render. Without styles the container
# keeps its `hidden` attribute and only the `<pre>` fallback shows; with styles
# the fallback is hidden and the container shown.
#
# Colours are deliberately not themed. Text inherits the notebook's colour and
# backgrounds and borders are translucent grey, so the diagram reads on a light
# or dark notebook alike. Keying a dark palette on `prefers-color-scheme` was
# tried and is wrong: that reports the operating system's setting, not the
# notebook's, so a dark OS with a light notebook drew dark boxes and invisible
# headings on a white page.
_CSS = """
#ROOT {
  --cm-border: rgba(128, 128, 128, 0.45);
  --cm-bg: rgba(128, 128, 128, 0.08);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.9em; color: inherit;
}
#ROOT .cm-fallback { display: none; }
#ROOT .cm-container { display: inline-block !important; }
#ROOT .cm-head { margin-bottom: 4px; }
#ROOT .cm-step {
  border: 1px solid var(--cm-border); border-left-width: 4px; border-radius: 6px;
  background: var(--cm-bg); padding: 4px 10px; min-width: 240px;
}
#ROOT .cm-transformer { border-left-color: #8250df; }
#ROOT .cm-learner { border-left-color: #2f81f7; }
#ROOT .cm-detector { border-left-color: #d29922; }
#ROOT summary { cursor: pointer; }
#ROOT .cm-kind {
  font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.04em;
  opacity: 0.65; margin-right: 0.5em;
}
#ROOT .cm-name { font-weight: 600; }
#ROOT .cm-shape { font-size: 0.85em; opacity: 0.65; margin-left: 0.5em; }
#ROOT .cm-arrow { text-align: center; opacity: 0.55; line-height: 1.3; }
#ROOT .cm-nested {
  border: 1px dashed var(--cm-border); border-radius: 6px; padding: 6px 8px;
}
#ROOT .cm-empty { opacity: 0.65; font-style: italic; }
#ROOT table { margin: 4px 0 2px; font-size: 0.85em; border-collapse: collapse; }
#ROOT td { padding: 1px 10px 1px 0; vertical-align: top; }
#ROOT td.cm-key { opacity: 0.65; }
"""


def render(pipeline: BasePipeline) -> str:
    """Return the HTML for ``pipeline``, styles included."""
    root_id = f"capymoa-pipeline-{next(_render_ids)}"
    text = f"{type(pipeline).__name__}: {str(pipeline) or 'no steps'}"
    return (
        f'<div id="{root_id}" class="cm-root">'
        f"<style>{_CSS.replace('#ROOT', '#' + root_id)}</style>"
        f'<pre class="cm-fallback">{html.escape(text)}</pre>'
        f'<div class="cm-container" hidden>{_render_pipeline(pipeline)}</div>'
        "</div>"
    )


def _render_pipeline(pipeline: BasePipeline) -> str:
    head = (
        '<div class="cm-head">'
        f'<span class="cm-name">{html.escape(type(pipeline).__name__)}</span>'
        f"{_shape(pipeline)}"
        "</div>"
    )
    if not pipeline.elements:
        return head + '<div class="cm-empty">no steps</div>'
    steps = '<div class="cm-arrow">&darr;</div>'.join(
        _render_step(element) for element in pipeline.elements
    )
    return head + f"<div>{steps}</div>"


def _render_step(element: PipelineElement) -> str:
    from .pipeline import BasePipeline

    if isinstance(element, BasePipeline):
        return f'<div class="cm-nested">{_render_pipeline(element)}</div>'

    css_class, kind, name, params, note = _describe(element)
    summary = (
        f'<span class="cm-kind">{html.escape(kind)}</span>'
        f'<span class="cm-name">{html.escape(name)}</span>'
        f"{note}"
    )
    if not params:
        return f'<div class="cm-step {css_class}">{summary}</div>'
    rows = "".join(
        f'<tr><td class="cm-key">{html.escape(str(key))}</td>'
        f"<td>{html.escape(_short(value))}</td></tr>"
        for key, value in params.items()
    )
    return (
        f'<details class="cm-step {css_class}">'
        f"<summary>{summary}</summary><table>{rows}</table></details>"
    )


def _describe(
    element: PipelineElement,
) -> tuple[str, str, str, dict[str, Any] | None, str]:
    """Return the CSS class, kind label, name, parameters and shape note."""
    from .pipeline import (
        ClassifierPipelineElement,
        DriftDetectorPipelineElement,
        RegressorPipelineElement,
        TransformerPipelineElement,
    )

    if isinstance(element, TransformerPipelineElement):
        return (
            "cm-transformer",
            "Transformer",
            _transformer_name(element.transformer),
            None,
            _shape(element),
        )
    if isinstance(element, (ClassifierPipelineElement, RegressorPipelineElement)):
        kind = (
            "Classifier"
            if isinstance(element, ClassifierPipelineElement)
            else "Regressor"
        )
        learner = element.learner
        # RandomSearchClassifierPE subclasses ClassifierPipelineElement but holds
        # its candidate models itself, so it has no single learner to show.
        name = str(element) if learner is None else str(learner)
        return "cm-learner", kind, name, _params(learner), _shape(element)
    if isinstance(element, DriftDetectorPipelineElement):
        monitor = _callable_name(element.prepare_drift_detector_input_func)
        note = f'<span class="cm-shape">monitors {html.escape(monitor)}</span>'
        return (
            "cm-detector",
            "Drift detector",
            str(element.drift_detector),
            _params(element.drift_detector),
            note,
        )
    return "", "Step", str(element), None, _shape(element)


def _shape(element: PipelineElement) -> str:
    """Describe the schema an element works on, or return an empty string."""
    incoming = element.get_input_schema()
    outgoing = element.get_schema()
    if incoming is not None and outgoing is not None:
        n_in, n_out = incoming.get_num_attributes(), outgoing.get_num_attributes()
        text = (
            f"{n_in} attributes"
            if n_in == n_out
            else f"{n_in} &rarr; {n_out} attributes"
        )
    elif outgoing is not None:
        text = f"{outgoing.get_num_attributes()} attributes"
    else:
        return ""
    return f'<span class="cm-shape">{text}</span>'


def _transformer_name(transformer: Any) -> str:
    """Name a transformer by its MOA filter and options, when it has one."""
    moa_filter = getattr(transformer, "moa_filter", None)
    if moa_filter is None:
        return str(transformer)
    return str(moa_filter.getCLICreationString(moa_filter.__class__)).strip()


def _callable_name(func: Any) -> str:
    """Name an input callable, unwrapping the monitor factories.

    ``prediction_is_correct()`` returns a closure whose qualified name is
    ``prediction_is_correct.<locals>.monitor``; the factory is the useful part.
    """
    qualname = getattr(func, "__qualname__", None) or type(func).__name__
    return qualname.split(".<locals>")[0]


def _params(obj: Any) -> dict[str, Any] | None:
    get_params = getattr(obj, "get_params", None)
    if get_params is None:
        return None
    try:
        params = get_params()
    except (TypeError, AttributeError, ValueError):
        return None
    return params if isinstance(params, dict) else None


def _short(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_VALUE_CHARS:
        return text[: _MAX_VALUE_CHARS - 3] + "..."
    return text
