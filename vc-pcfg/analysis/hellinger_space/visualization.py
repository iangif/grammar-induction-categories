"""Simple interactive single-model visualizations across linguistic feature spaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import FEATURE_SPACES, PROJECTIONS
from .io import HellingerSpaceError, write_csv
from .projections import project_pca, project_tsne, project_umap
from .summaries import active_feature_scores, merge_category_visual_metrics
from .transform import HellingerSpace, build_hellinger_space


def _projection_result(
    space: HellingerSpace,
    projection: str,
    *,
    random_state: int,
) -> dict[str, Any]:
    if projection == "pca":
        return project_pca(space)
    if projection == "umap":
        return project_umap(space, random_state=random_state)
    if projection == "tsne":
        return project_tsne(space, random_state=random_state)
    raise HellingerSpaceError(f"Unknown projection {projection!r}.")


def _xy_columns(projection: str, coordinates: pd.DataFrame) -> tuple[str, str]:
    mapping = {
        "pca": ("PC1", "PC2"),
        "umap": ("UMAP1", "UMAP2"),
        "tsne": ("TSNE1", "TSNE2"),
    }
    x, y = mapping[projection]
    if x not in coordinates.columns:
        raise HellingerSpaceError(f"Projection output is missing {x}.")
    if y not in coordinates.columns:
        # PCA can be 1-D for a two-category input. Keep the HTML operational.
        coordinates[y] = 0.0
    return x, y


def _pca_interpretation(result: dict[str, Any], *, top_n: int = 5) -> dict[str, Any]:
    explained = result["explained_variance"]
    contributions = result["feature_contributions"]
    top_loadings = result["top_loadings"]
    output: dict[str, Any] = {}
    for component in ("PC1", "PC2"):
        row = explained.loc[explained["component"].eq(component)]
        if row.empty:
            continue
        ratio = float(row.iloc[0]["explained_variance_ratio"])
        blocks = contributions.loc[contributions["component"].eq(component)].head(top_n)
        positive = top_loadings.loc[
            top_loadings["component"].eq(component) & top_loadings["side"].eq("positive")
        ].head(top_n)
        negative = top_loadings.loc[
            top_loadings["component"].eq(component) & top_loadings["side"].eq("negative")
        ].head(top_n)
        output[component] = {
            "variance": ratio,
            "blocks": [
                {
                    "position": str(item.position),
                    "feature": str(item.feature),
                    "family": str(item.feature_family),
                    "contribution": float(item.contribution),
                }
                for item in blocks.itertuples(index=False)
            ],
            "positive": [
                {
                    "position": str(item.position),
                    "feature": str(item.feature),
                    "value": str(item.value),
                    "loading": float(item.loading),
                }
                for item in positive.itertuples(index=False)
            ],
            "negative": [
                {
                    "position": str(item.position),
                    "feature": str(item.feature),
                    "value": str(item.value),
                    "loading": float(item.loading),
                }
                for item in negative.itertuples(index=False)
            ],
        }
    return output


def _record(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _category_detail_payload(
    active_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    hover: dict[str, list[dict[str, Any]]] = {}
    for category, group in active_scores.groupby("category", sort=False, dropna=False):
        rows: list[dict[str, Any]] = []
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "position": str(row.position),
                    "feature": str(row.feature),
                    "family": str(row.feature_family),
                    "modal_value": _record(row.modal_value),
                    "modal_coverage": float(row.modal_coverage),
                    "corpus_coverage": _record(row.corpus_coverage),
                    "coherence": float(row.normalized_coherence),
                    "block_weight": float(row.block_weight),
                }
            )
        key = str(category)
        details[key] = rows
        hover[key] = rows[:top_k]
    return details, hover


def _write_space_projection_outputs(
    output_dir: Path,
    *,
    space: HellingerSpace,
    visual_metrics: pd.DataFrame,
    projection: str,
    result: dict[str, Any],
) -> list[str]:
    space_dir = output_dir / space.feature_space
    space_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    common = {
        "category_hellinger_vectors.csv": space.vectors,
        "hellinger_dimensions.csv": space.dimensions,
        "space_category_summary.csv": visual_metrics,
    }
    for name, frame in common.items():
        path = space_dir / name
        if not path.exists():
            write_csv(frame, path)
            outputs.append(str(path.relative_to(output_dir)))

    coordinates_name = f"{projection}_coordinates.csv"
    coordinates_path = space_dir / coordinates_name
    write_csv(result["coordinates"], coordinates_path)
    outputs.append(str(coordinates_path.relative_to(output_dir)))

    if projection == "pca":
        pca_outputs = {
            "pca_loadings.csv": result["loadings"],
            "pca_explained_variance.csv": result["explained_variance"],
            "pca_feature_contributions.csv": result["feature_contributions"],
            "pca_top_loadings.csv": result["top_loadings"],
        }
        for name, frame in pca_outputs.items():
            path = space_dir / name
            write_csv(frame, path)
            outputs.append(str(path.relative_to(output_dir)))
    return outputs


def _json_for_script(payload: Any) -> str:
    # Prevent literal </script> in corpus-derived strings from ending the data script.
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _write_html(payload: dict[str, Any], path: Path) -> None:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise HellingerSpaceError(
            "Interactive HTML requires Plotly. Install it with `uv pip install plotly`."
        ) from exc

    plotly_js = get_plotlyjs()
    data_json = _json_for_script(payload)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Induced Category Hellinger Space</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
.controls {{ display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
label {{ font-size: 14px; }} select {{ margin-left: 5px; }}
#plot {{ width: 100%; height: 660px; }}
.meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 8px; }}
.panel {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; min-width: 0; }}
#detail-table-wrap {{ max-height: 360px; overflow-y: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ text-align: left; border-bottom: 1px solid #eee; padding: 5px 7px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: white; }}
.small {{ color: #666; font-size: 12px; }}
.pc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.pc-grid ul {{ margin-top: 4px; padding-left: 20px; }}
@media (max-width: 850px) {{ .meta, .pc-grid {{ grid-template-columns: 1fr; }} #plot {{ height: 520px; }} }}
</style>
<script>{plotly_js}</script>
</head>
<body>
<h2>Induced Category Hellinger Space</h2>
<div class="controls">
  <label>Feature space <select id="space-select"></select></label>
  <label>Projection <select id="projection-select"></select></label>
  <label><input id="labels-toggle" type="checkbox"> Show category labels</label>
</div>
<div id="plot"></div>
<div class="meta">
  <div class="panel">
    <h3 style="margin-top:0">Projection interpretation</h3>
    <div id="interpretation" class="small"></div>
  </div>
  <div class="panel">
    <h3 style="margin-top:0">Category details</h3>
    <div id="category-summary" class="small">Click a point to inspect all active features.</div>
    <div id="detail-table-wrap"></div>
  </div>
</div>
<script id="payload" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const plot = document.getElementById('plot');
const spaceSelect = document.getElementById('space-select');
const projectionSelect = document.getElementById('projection-select');
const labelsToggle = document.getElementById('labels-toggle');

function escapeHtml(x) {{
  return String(x ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}})[c]);
}}
function pct(x) {{ return (100 * x).toFixed(1) + '%'; }}
function num(x) {{ return (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(3); }}

for (const space of DATA.spaces) {{
  const option = document.createElement('option'); option.value = space; option.textContent = DATA.space_labels[space]; spaceSelect.appendChild(option);
}}
for (const projection of DATA.projections) {{
  const option = document.createElement('option'); option.value = projection; option.textContent = DATA.projection_labels[projection]; projectionSelect.appendChild(option);
}}

function renderInterpretation(view) {{
  const target = document.getElementById('interpretation');
  let quality = Object.entries(view.quality).map(([k,v]) => `${{escapeHtml(k.replaceAll('_',' '))}}: <b>${{num(v)}}</b>`).join(' &nbsp; ');
  if (view.projection !== 'pca') {{
    target.innerHTML = `<div>${{quality}}</div><p>Nonlinear coordinates are exploratory; inspect quality metrics before interpreting 2-D neighborhoods.</p>`;
    return;
  }}
  const pcs = view.pca_interpretation || {{}};
  const chunks = [];
  for (const pc of ['PC1','PC2']) {{
    if (!pcs[pc]) continue;
    const info = pcs[pc];
    const blocks = info.blocks.map(x => `<li>${{escapeHtml(x.position)}} · ${{escapeHtml(x.feature)}} (${{pct(x.contribution)}})</li>`).join('');
    const pos = info.positive.map(x => `<li>${{escapeHtml(x.position)}} · ${{escapeHtml(x.feature)}} = ${{escapeHtml(x.value)}} (${{Number(x.loading).toFixed(3)}})</li>`).join('');
    const neg = info.negative.map(x => `<li>${{escapeHtml(x.position)}} · ${{escapeHtml(x.feature)}} = ${{escapeHtml(x.value)}} (${{Number(x.loading).toFixed(3)}})</li>`).join('');
    chunks.push(`<div><b>${{pc}} — ${{pct(info.variance)}} variance</b><div>Top feature blocks:</div><ul>${{blocks}}</ul><div>+ side:</div><ul>${{pos}}</ul><div>− side:</div><ul>${{neg}}</ul></div>`);
  }}
  target.innerHTML = `<div>${{quality}}</div><div class="pc-grid">${{chunks.join('')}}</div><p>PCA signs are arbitrary; interpret each axis as the contrast between its two ends.</p>`;
}}

function renderDetails(category) {{
  const space = spaceSelect.value;
  const summary = DATA.category_metrics[String(category)];
  const rows = DATA.details[space][String(category)] || [];
  let header = `<b>Category ${{escapeHtml(category)}}</b> &nbsp; LD_eff: <b>${{num(summary.LD_eff)}}</b> &nbsp; ${{escapeHtml(DATA.space_labels[space])}} coherence: <b>${{num(summary.space_coherence_by_space[space])}}</b>`;
  if (summary.LC !== null && summary.LC !== undefined) header += ` &nbsp; LC: <b>${{num(summary.LC)}}</b>`;
  if (summary.CC2 !== null && summary.CC2 !== undefined) header += ` &nbsp; CC2: <b>${{num(summary.CC2)}}</b>`;
  document.getElementById('category-summary').innerHTML = header;
  let table = '<table><thead><tr><th>Position</th><th>Feature</th><th>Modal value</th><th>Coverage</th><th>Corpus</th><th>Coherence</th></tr></thead><tbody>';
  for (const r of rows) {{
    table += `<tr><td>${{escapeHtml(r.position)}}</td><td>${{escapeHtml(r.feature)}}</td><td>${{escapeHtml(r.modal_value ?? 'NULL')}}</td><td>${{num(r.modal_coverage)}}</td><td>${{num(r.corpus_coverage)}}</td><td>${{num(r.coherence)}}</td></tr>`;
  }}
  table += '</tbody></table>';
  document.getElementById('detail-table-wrap').innerHTML = table;
}}

function render() {{
  const space = spaceSelect.value;
  const projection = projectionSelect.value;
  const view = DATA.views[space][projection];
  const mode = labelsToggle.checked ? 'markers+text' : 'markers';
  const trace = {{
    type: 'scatter', mode,
    x: view.x, y: view.y, text: view.categories,
    textposition: 'top center',
    customdata: view.categories,
    hovertext: view.hovertext, hoverinfo: 'text',
    marker: {{
      size: view.ld_eff, sizemode: 'area', sizeref: DATA.sizeref, sizemin: 5,
      color: view.coherence, cmin: 0, cmax: 1, colorscale: 'Viridis',
      colorbar: {{title: 'Coherence'}}, line: {{width: 0.5, color: '#555'}}
    }}
  }};
  const layout = {{
    title: `${{DATA.space_labels[space]}} feature space — ${{DATA.projection_labels[projection]}}`,
    xaxis: {{title: view.x_label, zeroline: projection === 'pca', range: view.x_range || undefined}},
    yaxis: {{title: view.y_label, zeroline: projection === 'pca', range: view.y_range || undefined, scaleanchor: projection === 'pca' ? 'x' : undefined, scaleratio: 1}},
    hovermode: 'closest', margin: {{l: 70, r: 45, t: 55, b: 65}}
  }};
  Plotly.react(plot, [trace], layout, {{responsive: true, displaylogo: false}});
  renderInterpretation(view);
}}

render();
plot.on('plotly_click', ev => {{ if (ev.points && ev.points.length) renderDetails(ev.points[0].customdata); }});
spaceSelect.addEventListener('change', () => {{ render(); document.getElementById('detail-table-wrap').innerHTML=''; document.getElementById('category-summary').textContent='Click a point to inspect all active features.'; }});
projectionSelect.addEventListener('change', render);
labelsToggle.addEventListener('change', render);
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def build_interactive_visualization(
    *,
    source: pd.DataFrame,
    category_metrics: pd.DataFrame,
    feature_scores: pd.DataFrame,
    output_dir: Path,
    spaces: Iterable[str] = FEATURE_SPACES,
    projections: Iterable[str] = ("pca", "umap"),
    position_group_weights: Mapping[str, float] | None = None,
    top_k: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    """Build all selected single-model views and one simple interactive HTML."""

    space_names = list(dict.fromkeys(spaces))
    projection_names = list(dict.fromkeys(projections))
    invalid_spaces = [name for name in space_names if name not in FEATURE_SPACES]
    invalid_projections = [name for name in projection_names if name not in PROJECTIONS]
    if invalid_spaces:
        raise HellingerSpaceError(f"Unknown feature spaces: {invalid_spaces}")
    if invalid_projections:
        raise HellingerSpaceError(f"Unknown projections: {invalid_projections}")
    if not space_names or not projection_names:
        raise HellingerSpaceError("At least one feature space and one projection are required.")
    if top_k <= 0:
        raise HellingerSpaceError("top_k must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "spaces": space_names,
        "projections": projection_names,
        "space_labels": {
            "all": "All",
            "lexical": "Lexical",
            "contextual": "Contextual",
            "grammatical": "Grammatical / morphosyntactic",
            "semantic": "Semantic",
        },
        "projection_labels": {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"},
        "views": {},
        "details": {},
        "category_metrics": {},
    }
    outputs: list[str] = []
    quality_rows: list[dict[str, Any]] = []
    all_space_coherence: dict[str, dict[str, float]] = {}

    max_ld = float(category_metrics["LD_eff"].max())
    payload["sizeref"] = 2.0 * max(max_ld, 1.0) / (42.0**2)

    for space_name in space_names:
        space = build_hellinger_space(
            source,
            feature_space=space_name,
            position_group_weights=position_group_weights,
        )
        visual_metrics = merge_category_visual_metrics(space, category_metrics, feature_scores)
        active_scores = active_feature_scores(space, feature_scores)
        details, hover_rows = _category_detail_payload(active_scores, visual_metrics, top_k=top_k)
        payload["details"][space_name] = details
        payload["views"][space_name] = {}
        all_space_coherence[space_name] = {
            str(row.category): float(row.space_coherence)
            for row in visual_metrics.itertuples(index=False)
        }

        for projection in projection_names:
            result = _projection_result(space, projection, random_state=random_state)
            outputs.extend(
                _write_space_projection_outputs(
                    output_dir,
                    space=space,
                    visual_metrics=visual_metrics,
                    projection=projection,
                    result=result,
                )
            )
            coordinates = result["coordinates"].copy()
            x_col, y_col = _xy_columns(projection, coordinates)
            merged = coordinates[["category", x_col, y_col]].merge(
                visual_metrics,
                on="category",
                how="inner",
                validate="one_to_one",
            )

            if projection == "pca":
                explained = result["explained_variance"].set_index("component")
                x_ratio = float(explained.loc["PC1", "explained_variance_ratio"])
                y_ratio = (
                    float(explained.loc["PC2", "explained_variance_ratio"])
                    if "PC2" in explained.index
                    else 0.0
                )
                x_label = f"PC1 ({100 * x_ratio:.1f}%)"
                y_label = f"PC2 ({100 * y_ratio:.1f}%)"
                interpretation = _pca_interpretation(result)
            else:
                prefix = "UMAP" if projection == "umap" else "t-SNE"
                x_label = f"{prefix} 1"
                y_label = f"{prefix} 2"
                interpretation = None

            hover_text: list[str] = []
            for row in merged.itertuples(index=False):
                category = str(row.category)
                lines = [
                    f"<b>Category {category}</b>",
                    f"LD_eff: {float(row.LD_eff):.2f}",
                    f"{space_name} coherence: {float(row.space_coherence):.3f}",
                    "<br><b>Top features</b>",
                ]
                for item in hover_rows.get(category, []):
                    corpus = item["corpus_coverage"]
                    corpus_text = "—" if corpus is None else f"{float(corpus):.3f}"
                    lines.append(
                        f"{item['position']} · {item['feature']} = {item['modal_value'] if item['modal_value'] not in (None, '') else 'NULL'} "
                        f"| coh {item['coherence']:.3f} | cov {item['modal_coverage']:.3f} "
                        f"| corpus {corpus_text}"
                    )
                hover_text.append("<br>".join(lines))

            view = {
                "projection": projection,
                "categories": [str(value) for value in merged["category"].tolist()],
                "x": merged[x_col].astype(float).tolist(),
                "y": merged[y_col].astype(float).tolist(),
                "ld_eff": merged["LD_eff"].astype(float).tolist(),
                "coherence": merged["space_coherence"].astype(float).tolist(),
                "hovertext": hover_text,
                "x_label": x_label,
                "y_label": y_label,
                "quality": {key: float(value) for key, value in result["quality"].items()},
                "pca_interpretation": interpretation,
            }
            payload["views"][space_name][projection] = view
            for metric, value in result["quality"].items():
                quality_rows.append(
                    {
                        "feature_space": space_name,
                        "projection": projection,
                        "metric": metric,
                        "value": float(value),
                    }
                )

    # Keep every PCA subspace on one common numerical/visual scale. The PC axes
    # have different linguistic meanings across subspaces, but a unit of
    # weighted-Hellinger coordinate distance is still the same unit. UMAP and
    # t-SNE are intentionally left auto-ranged because their coordinate scales
    # are arbitrary across independently fitted spaces.
    if "pca" in projection_names:
        pca_values: list[float] = []
        for space_name in space_names:
            view = payload["views"][space_name].get("pca")
            if view is not None:
                pca_values.extend(abs(float(value)) for value in view["x"])
                pca_values.extend(abs(float(value)) for value in view["y"])
        limit = max(pca_values, default=1.0)
        limit = max(limit * 1.08, 1e-9)
        for space_name in space_names:
            view = payload["views"][space_name].get("pca")
            if view is not None:
                view["x_range"] = [-limit, limit]
                view["y_range"] = [-limit, limit]

    metric_columns = [column for column in ("LC", "CC2") if column in category_metrics.columns]
    for row in category_metrics.itertuples(index=False):
        key = str(row.category)
        entry: dict[str, Any] = {
            "LD_eff": float(row.LD_eff),
            "space_coherence_by_space": {
                space: all_space_coherence[space][key] for space in space_names
            },
        }
        for column in metric_columns:
            entry[column] = _record(getattr(row, column))
        payload["category_metrics"][key] = entry

    quality = pd.DataFrame(quality_rows)
    quality_path = output_dir / "projection_quality.csv"
    write_csv(quality, quality_path)
    outputs.append(quality_path.name)

    html_path = output_dir / "interactive_category_map.html"
    _write_html(payload, html_path)
    outputs.append(html_path.name)

    return {
        "spaces": space_names,
        "projections": projection_names,
        "quality": quality,
        "outputs": list(dict.fromkeys(outputs)),
    }
