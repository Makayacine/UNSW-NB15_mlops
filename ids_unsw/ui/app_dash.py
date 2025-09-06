import os, io, base64, json, requests
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, dash_table, no_update
import dash_bootstrap_components as dbc

# ----- Config (prefilled from env like before) -----
DEFAULT_API = os.getenv("IDS_API_URL", "http://host.docker.internal:8000")
DEFAULT_TOKEN = os.getenv("IDS_API_TOKEN", "")

def _h(base_url, token):
    return {
        "base": (base_url or DEFAULT_API).rstrip("/"),
        "headers": {"Authorization": f"Bearer {token or DEFAULT_TOKEN}"},
        "timeout": 10,
    }

def _get(base_url, token, path):
    h = _h(base_url, token)
    r = requests.get(h["base"] + path, headers=h["headers"], timeout=h["timeout"])
    r.raise_for_status()
    return r.json()

def _post(base_url, token, path, payload):
    h = _h(base_url, token)
    r = requests.post(h["base"] + path, headers={**h["headers"], "Content-Type": "application/json"},
                      json=payload, timeout=h["timeout"])
    r.raise_for_status()
    return r.json()

# ============= App =============
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], title="UNSW IDS – Dash")
server = app.server

# Stores to cache server state
stores = html.Div([
    dcc.Store(id="store-features", data=[]),
    dcc.Store(id="store-threshold", data=None),
])

# Left connection panel
left_panel = dbc.Card(
    dbc.CardBody([
        html.H6("Connection", className="text-muted"),
        dbc.Label("API base URL"),
        dbc.Input(id="in-base", value=DEFAULT_API, placeholder="http://host.docker.internal:8000"),
        dbc.Label("API token (Bearer)", className="mt-2"),
        dbc.Input(id="in-token", value=DEFAULT_TOKEN, type="password"),
        dbc.Button("Ping /health", id="btn-ping", color="primary", className="mt-3", n_clicks=0),
        html.Div(id="ping-status", className="mt-2 small"),
        html.Hr(),
        html.P("Tip: set env vars IDS_API_URL and IDS_API_TOKEN to prefill.", className="text-muted small")
    ]),
    className="mb-3"
)

# Dynamic form container (built after ping → /features)
feature_form = html.Div(id="feature-form", children=[
    html.Div("Click 'Ping /health' to load feature schema.", className="text-muted")
])

single_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                id="endpoint-choice",
                options=[{"label": "/predict (label)", "value": "predict"},
                         {"label": "/predict_proba (prob only)", "value": "predict_proba"}],
                value="predict",
                inline=True,
                className="mb-3"
            ),
            feature_form,
            dbc.Button("Predict", id="btn-predict", color="success", className="mt-3"),
            html.Div(id="single-result", className="mt-3")
        ], width=12)
    ])
], fluid=True)

batch_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                id="endpoint-batch",
                options=[{"label": "/predict", "value": "predict"},
                         {"label": "/predict_proba", "value": "predict_proba"}],
                value="predict",
                inline=True,
                className="mb-2"
            ),
            dcc.Upload(
                id="upload-csv",
                children=html.Div(["Drag & drop CSV here or ", html.A("choose file")]),
                multiple=False,
                style={
                    "width": "100%", "height": "80px", "lineHeight": "80px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "6px", "textAlign": "center", "margin": "10px 0"
                },
            ),
            dbc.Button("Score file", id="btn-score", color="secondary", className="mb-3"),
            html.Div(id="batch-msg", className="mb-2"),
            dash_table.DataTable(
                id="batch-table", page_size=10, style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "monospace", "fontSize": 13}
            ),
            dcc.Download(id="dl-results")
        ], width=12)
    ])
], fluid=True)

admin_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H5("Admin / Ops", className="mb-3"),
            dbc.InputGroup([
                dbc.InputGroupText("New threshold"),
                dbc.Input(id="in-thr", type="number", min=0, max=1, step=0.0001, placeholder="0.6285714"),
                dbc.Button("Apply", id="btn-set-thr", color="warning")
            ]),
            dbc.Button("Reload model", id="btn-reload", color="info", className="ms-2"),
            html.Div(id="admin-msg", className="mt-3 small text-muted")
        ], width=12)
    ])
], fluid=True)

tabs = dcc.Tabs(id="main-tabs", value="single", children=[
    dcc.Tab(label="Single prediction", value="single", children=single_tab),
    dcc.Tab(label="Batch scoring", value="batch", children=batch_tab),
    dcc.Tab(label="Admin / Ops", value="admin", children=admin_tab),
])

app.layout = dbc.Container([
    stores,
    dbc.Row([
        dbc.Col(left_panel, width=3),
        dbc.Col(tabs, width=9),
    ], className="mt-3")
], fluid=True)

# ---------- Callbacks ----------

@app.callback(
    Output("ping-status", "children"),
    Output("store-features", "data"),
    Output("store-threshold", "data"),
    Input("btn-ping", "n_clicks"),
    State("in-base", "value"),
    State("in-token", "value"),
    prevent_initial_call=True
)
def ping(n, base, token):
    try:
        h = _get(base, token, "/health")
        feats = _get(base, token, "/features")["features"]
        thr = float(h.get("threshold", 0.5))
        return (dbc.Alert(f"OK – {len(feats)} features; threshold={thr:.6f}", color="success", dismissable=True),
                feats, thr)
    except Exception as e:
        return (dbc.Alert(f"Ping failed: {e}", color="danger", dismissable=True), [], None)

@app.callback(
    Output("feature-form", "children"),
    Input("store-features", "data")
)
def build_form(features):
    if not features:
        return html.Div("No features loaded yet.", className="text-muted")
    # Build 2-column compact numeric form
    rows = []
    for i in range(0, len(features), 2):
        row_inputs = []
        for f in features[i:i+2]:
            row_inputs.append(
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText(f),
                    dbc.Input(id={"type": "feat-input", "index": f}, type="number",
                              step=0.0001, value=0)
                ]), width=6)
            )
        rows.append(dbc.Row(row_inputs, className="mb-2"))
    return rows

@app.callback(
    Output("single-result", "children"),
    Input("btn-predict", "n_clicks"),
    State("endpoint-choice", "value"),
    State("store-features", "data"),
    State({"type": "feat-input", "index": ALL}, "value"),
    State({"type": "feat-input", "index": ALL}, "id"),
    State("in-base", "value"),
    State("in-token", "value"),
    State("store-threshold", "data"),
    prevent_initial_call=True
)
def do_single(n, endpoint, features, values, ids, base, token, thr):
    if not features:
        return dbc.Alert("No features loaded. Click Ping first.", color="warning")
    row = {}
    for v, meta in zip(values, ids):
        row[meta["index"]] = float(v or 0)
    try:
        payload = {"instances": [row]}
        res = _post(base, token, f"/{endpoint}", payload)
        if endpoint == "predict":
            p = float(res["probabilities"][0])
            y = int(res["predictions"][0])
            t = float(res["threshold"])
            return dbc.Alert(f"prob={p:.6f}  |  pred={y}  |  thr={t:.6f}", color="success")
        else:
            p = float(res["probabilities"][0])
            # If we know thr, compute label for display:
            y = int(p >= (thr if thr is not None else 0.5))
            return dbc.Alert(f"prob={p:.6f}  |  pred≈{y}", color="info")
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}", color="danger")

def _parse_upload(contents):
    # contents: "data:text/csv;base64,...."
    if not contents:
        return None
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df

@app.callback(
    Output("batch-msg", "children"),
    Output("batch-table", "data"),
    Output("batch-table", "columns"),
    Output("dl-results", "data"),
    Input("btn-score", "n_clicks"),
    State("upload-csv", "contents"),
    State("store-features", "data"),
    State("endpoint-batch", "value"),
    State("in-base", "value"),
    State("in-token", "value"),
    prevent_initial_call=True
)
def do_batch(n, contents, features, endpoint, base, token):
    if not contents:
        return (dbc.Alert("Please upload a CSV first.", color="warning"), [], [], no_update)
    if not features:
        return (dbc.Alert("No features loaded. Click Ping first.", color="warning"), [], [], no_update)
    try:
        df = _parse_upload(contents)
        missing = [f for f in features if f not in df.columns]
        extra = [c for c in df.columns if c not in features]
        if missing:
            return (dbc.Alert(f"Missing columns: {missing[:8]}...", color="danger"), [], [], no_update)

        instances = df[features].astype(float).to_dict(orient="records")
        res = _post(base, token, f"/{endpoint}", {"instances": instances})

        if endpoint == "predict":
            df_out = df.copy()
            df_out["probability"] = res["probabilities"]
            df_out["prediction"] = res["predictions"]
        else:
            df_out = df.copy()
            df_out["probability"] = res["probabilities"]

        cols = [{"name": c, "id": c} for c in df_out.columns]
        msg = dbc.Alert(f"Scored {len(df_out)} rows. Extra cols ignored: {extra[:8]}", color="success")
        # Download CSV
        return (msg, df_out.to_dict("records"), cols,
                dict(content=df_out.to_csv(index=False), filename="batch_scored.csv"))
    except Exception as e:
        return (dbc.Alert(f"Batch failed: {e}", color="danger"), [], [], no_update)

@app.callback(
    Output("admin-msg", "children"),
    Input("btn-set-thr", "n_clicks"),
    Input("btn-reload", "n_clicks"),
    State("in-thr", "value"),
    State("in-base", "value"),
    State("in-token", "value"),
    prevent_initial_call=True
)
def admin_actions(n_thr, n_reload, thr, base, token):
    trig = ctx.triggered_id
    try:
        if trig == "btn-set-thr":
            if thr is None:
                return dbc.Alert("Enter a numeric threshold.", color="warning")
            res = _post(base, token, "/set_threshold", {"threshold": float(thr)})
            return dbc.Alert(f"Threshold applied. Live thr={res['threshold']:.6f}", color="success")
        else:
            res = _post(base, token, "/reload", {})
            return dbc.Alert(f"Reloaded OK (features={res['n_features']})", color="info")
    except Exception as e:
        return dbc.Alert(f"Admin call failed: {e}", color="danger")

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
