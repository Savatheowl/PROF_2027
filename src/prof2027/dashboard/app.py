"""Веб-дашборд PROF_2027 (Dash).

Роли: admin — все вкладки, analyst — без администрирования,
viewer — read-only (обзор, материалы, предметы, датасет, модели).
Доступы: admin/admin, analyst/analyst, viewer/viewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import Input, Output, State, dash_table, dcc, html, no_update
from flask import session

from prof2027.config import Config
from prof2027.dashboard import data as ddata
from prof2027.db import User, get_engine, init_db, make_session
from prof2027.security import verify_password

ROLE_TABS = {
    "admin": ["overview", "materials", "subjects", "dataset", "cluster",
              "models", "drift", "trajectories", "reports", "admin"],
    "analyst": ["overview", "materials", "subjects", "dataset", "cluster",
                "models", "drift", "trajectories", "reports"],
    "viewer": ["overview", "materials", "subjects", "dataset", "models"],
}
TAB_LABELS = {
    "overview": "Обзор", "materials": "Материалы", "subjects": "Предметы",
    "dataset": "Датасет", "cluster": "Кластеризация", "models": "Обучение",
    "drift": "Дрейф", "trajectories": "Траектории", "reports": "Отчёты",
    "admin": "Администрирование",
}


def _engine(cfg):
    engine = get_engine(cfg.db_path)
    init_db(engine)
    return engine


def make_dashboard(cfg: Config | None = None):
    cfg = cfg or Config()
    app = dash.Dash(__name__, title="PROF_2027 — дашборд", update_title=None)
    server = app.server
    server.secret_key = "prof2027-demo-secret"

    @server.route("/reports/<path:fp>")
    def serve_reports(fp):
        from flask import send_from_directory

        return send_from_directory(str(cfg.root / "reports"), fp)

    # ------------------------------------------------------------------ #
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page"),
    ])

    # ---- страница: логин или основная ----
    @app.callback(Output("page", "children"), Input("url", "pathname"))
    def render(pathname):
        if not session.get("user"):
            return _login_layout()
        return _main_layout(session.get("role", "viewer"))

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("login-error", "children"),
        Input("login-btn", "n_clicks"),
        State("login-user", "value"),
        State("login-pass", "value"),
        prevent_initial_call=True,
    )
    def do_login(n, username, password):
        if not n:
            return no_update, no_update
        engine = _engine(cfg)
        with make_session(engine)() as s:
            users = {u.username: u for u in s.query(User).all()}
        u = users.get(username or "")
        if u and verify_password(password or "", u.password_hash):
            session["user"] = u.username
            session["role"] = u.role
            return "/", ""
        return no_update, "Неверный логин или пароль"

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("logout-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def do_logout(n):
        session.clear()
        return "/login"

    # ---- переключение вкладок ----
    @app.callback(Output("tab-content", "children"),
                  Input("tabs", "active_tab"))
    def render_tab(tab):
        engine = _engine(cfg)
        with make_session(engine)() as s:
            return _tab_content(tab, s, cfg, session.get("role", "viewer"))

    return app


# --------------------------------------------------------------------------- #
def _login_layout():
    return html.Div([
        html.H2("PROF_2027", style={"textAlign": "center", "marginTop": "8%"}),
        html.Div([
            dcc.Input(id="login-user", type="text", placeholder="логин", style=_inp()),
            html.Br(),
            dcc.Input(id="login-pass", type="password", placeholder="пароль", style=_inp()),
            html.Br(),
            html.Button("Войти", id="login-btn", n_clicks=0, style=_btn()),
            html.Div(id="login-error", style={"color": "red", "textAlign": "center", "marginTop": 8}),
        ], style={"maxWidth": 320, "margin": "0 auto"}),
        html.Div("admin/admin · analyst/analyst · viewer/viewer",
                 style={"textAlign": "center", "marginTop": 12, "color": "#888"}),
    ])


def _main_layout(role: str):
    tabs = ROLE_TABS.get(role, ROLE_TABS["viewer"])
    tab_children = [
        dcc.Tab(label=TAB_LABELS[t], value=t, id=f"tab-{t}") for t in tabs
    ]
    return html.Div([
        html.Div([
            html.H3(f"PROF_2027 · дашборд", style={"display": "inline-block"}),
            html.Span(f" [{session.get('user')} / {role}]",
                      style={"marginLeft": 12, "color": "#666"}),
            html.Button("Выход", id="logout-btn", n_clicks=0,
                        style={"float": "right", "marginTop": 18}),
        ], style={"padding": "6px 18px", "borderBottom": "1px solid #ddd"}),
        dcc.Tabs(id="tabs", value=tabs[0], children=tab_children),
        html.Div(id="tab-content", style={"padding": 14}),
    ])


def _tab_content(tab, s, cfg, role):
    if tab == "overview":
        return _tab_overview(s)
    if tab == "materials":
        return _tab_materials(s)
    if tab == "subjects":
        return _tab_subjects(s)
    if tab == "dataset":
        return _tab_dataset(s, cfg)
    if tab == "cluster":
        return _tab_cluster(s, cfg)
    if tab == "models":
        return _tab_models(s)
    if tab == "drift":
        return _tab_drift(s)
    if tab == "trajectories":
        return _tab_trajectories(s, cfg)
    if tab == "reports":
        return _tab_reports(cfg)
    if tab == "admin":
        return _tab_admin(s)
    return html.Div("—")


# --------------------------------------------------------------------------- #
def _tab_overview(s):
    k = ddata.kpis(s)
    cards = [
        ("Материалов", k["materials"]), ("Предметов", k["subjects"]),
        ("Категорий", k["categories"]), ("Принято", k["approved"]),
        ("На проверке", k["needs_review"]), ("Отклонено", k["rejected"]),
        ("Часов изучения", k["total_hours"]),
    ]
    row = html.Div([_kpi_card(t, v) for t, v in cards],
                   style={"display": "flex", "flexWrap": "wrap", "gap": 12})
    return html.Div([html.H4("Обзор"), row])


def _tab_materials(s):
    rows = ddata.materials_rows(s)
    cols = [{"name": c, "id": c} for c in
            ["id", "title", "subject", "level", "type", "words",
             "complexity", "methodology", "verdict", "hours"]]
    return html.Div([
        html.H4("Материалы"),
        dash_table.DataTable(rows, cols, page_size=25, style_table={"overflowX": "auto"},
                             style_cell={"fontSize": 13}),
    ])


def _tab_subjects(s):
    counts = ddata.subject_counts(s)
    labels = [c["subject"] for c in counts]
    values = [c["count"] for c in counts]
    fig = {
        "data": [{"type": "bar", "x": labels, "y": values, "name": "Материалы"}],
        "layout": {"title": "Материалы по предметам", "xaxis": {"tickangle": -35}},
    }
    return html.Div([html.H4("Предметы"), dcc.Graph(figure=fig)])


def _tab_dataset(s, cfg):
    means = ddata.compliance_means(s)
    labels = [m["requirement"][:30] for m in means]
    values = [m["mean"] for m in means]
    fig = {"data": [{"type": "bar", "x": labels, "y": values}],
           "layout": {"title": "Среднее соответствие по требованиям",
                      "yaxis": {"range": [0, 1]}, "xaxis": {"tickangle": -90}}}
    charts = ddata.chart_paths(cfg)
    imgs = [html.Img(src=f"/reports/{p}", style={"height": 220, "margin": 6})
            for p in charts if p.endswith(".png")]
    return html.Div([html.H4("Датасет"), dcc.Graph(figure=fig), html.Div(imgs)])


def _tab_cluster(s, cfg):
    runs = ddata.cluster_runs(s)
    cols = [{"name": c, "id": c} for c in ["id", "strategy", "algorithm",
                                           "n_clusters", "silhouette", "created"]]
    charts = ddata.chart_paths(cfg)
    imgs = [html.Img(src=f"/reports/{p}", style={"height": 260, "margin": 6})
            for p in charts if "tsne" in p]
    return html.Div([
        html.H4("Кластеризация"),
        dash_table.DataTable(runs, cols, page_size=10, style_cell={"fontSize": 13}),
        html.Div(imgs),
    ])


def _tab_models(s):
    rows = ddata.model_versions(s)
    cols = [{"name": c, "id": c} for c in
            ["id", "task", "algorithm", "version", "roc_auc", "accuracy",
             "r2", "rmse", "drift", "active"]]
    return html.Div([
        html.H4("Модели и метрики (GroupKFold по предметам)"),
        dash_table.DataTable(rows, cols, page_size=20, style_cell={"fontSize": 13}),
    ])


def _tab_drift(s):
    rows = ddata.model_versions(s)
    cols = [{"name": c, "id": c} for c in
            ["id", "task", "algorithm", "version", "drift", "active"]]
    return html.Div([
        html.H4("Дрейф данных (continuous check)"),
        dash_table.DataTable(rows, cols, page_size=20, style_cell={"fontSize": 13}),
        html.P("Запустите: prof2027 continuous check — обновит уровень дрейфа.",
               style={"color": "#666"}),
    ])


def _tab_trajectories(s, cfg):
    rows = ddata.trajectories(s)
    cols = [{"name": c, "id": c} for c in
            ["id", "user", "days", "hours", "deadline", "in_time", "created"]]
    gantts = sorted((cfg.root / "reports" / "predict").glob("gantt_*.png"),
                    key=lambda p: p.stat().st_mtime, reverse=True) if (cfg.root / "reports" / "predict").exists() else []
    imgs = [html.Img(src=f"/reports/predict/{p.name}", style={"height": 260, "margin": 6})
            for p in gantts[:2]]
    return html.Div([
        html.H4("Траектории"),
        dash_table.DataTable(rows, cols, page_size=10, style_cell={"fontSize": 13}),
        html.Div(imgs),
    ])


def _tab_reports(cfg):
    files = ddata.report_files(cfg)
    items = [html.Li(html.A(f["name"], href=f"/reports/{f['path']}", target="_blank"))
             for f in files]
    return html.Div([html.H4("Отчёты"), html.Ul(items)])


def _tab_admin(s):
    rows = ddata.users(s)
    cols = [{"name": c, "id": c} for c in ["id", "username", "role"]]
    return html.Div([
        html.H4("Администрирование: пользователи"),
        dash_table.DataTable(rows, cols, page_size=10, style_cell={"fontSize": 13}),
        html.P("Пароли по умолчанию: admin/admin, analyst/analyst, viewer/viewer. "
               "Меняются через .env (PROF2027_ADMIN_PASSWORD и др.).",
               style={"color": "#666", "marginTop": 10}),
    ])


# --------------------------------------------------------------------------- #
def _kpi_card(title, value):
    return html.Div([
        html.Div(title, style={"fontSize": 12, "color": "#888"}),
        html.Div(str(value), style={"fontSize": 26, "fontWeight": 600}),
    ], style={"border": "1px solid #ddd", "borderRadius": 8, "padding": "10px 18px",
              "minWidth": 120})


def _inp():
    return {"width": "100%", "padding": 8, "marginBottom": 8, "fontSize": 15}


def _btn():
    return {"width": "100%", "padding": 8, "fontSize": 15, "cursor": "pointer"}
