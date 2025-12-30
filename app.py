# app.py
# JAMS Investment Network Platform (Streamlit) — FULL CODE (REFRESHED)
# Home visualizations (ONLY):
# - Alumni by Country (interactive choropleth; tech-styled; no white border)
# - Top Firms (Top 3 + Others)
# - Alumni by Major (Top 5 + Others; donut; #921515 themed)

import io
import os
import re
import sqlite3
import time
import urllib.parse
from datetime import datetime, date
from typing import List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# =========================
# BRAND / CONFIG
# =========================
BRAND_RED = "#921515"
BRAND_GRAY_BG = "#F5F5F7"
BRAND_GRAY_BORDER = "#E8E8EC"
TEXT_DARK = "#1F1F23"

# Red palette for charts (primary + lighter reds + neutral gray)
JAMS_RED_SCALE = [
    "#921515",  # primary
    "#B65B5B",
    "#C97E7E",
    "#E3BDBD",
    "#F6EDED",
    "#B9BAC3",  # neutral gray for "Others" if desired
]

SHEET_URL = "https://docs.google.com/spreadsheets/d/17Uay_e3sySP8wkkwXKglS-WOn14UfxE7WLx5opDOXEs/edit?usp=sharing"
SHEET_ID = SHEET_URL.split("/d/")[1].split("/")[0]

TAB_ALUMNI = "Alumni"
TAB_RESOURCES = "Resources"
TAB_PASSWORDS = "Passwords"

DB_PATH = "jams_platform.db"

CAREER_SERVICES = ["Model Review", "Mock Interview", "Resume Review"]

HOME_INTRO = (
    "JAMS Investment Network Platform is designed to connect current members with the most relevant alumni from our network of 70+ professionals. "
    "Whether for career development or skill-building, the platform facilitates a wide range of professional interactions—including coffee chats, "
    "mock interviews, financial model reviews, and more—empowering members to leverage alumni expertise for any professional purpose."
)

ALUMNI_HEADERS = [
    "English Name",
    "Chinese Name",
    "Experience",
    "Cohort Year",
    "Email",
    "Linkedin",
    "JAMS Department",
    "School",
    "Major",
    "Location",
]
RES_HEADERS = ["Title", "Description", "Link"]
PWD_HEADERS = ["Name", "Password", "Status"]

# Plotly interactivity config (forces modebar + scroll zoom)
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}

# =========================
# LOGO HANDLING (Cloud-safe)
# =========================
LOGO_CANDIDATES = [
    "jams_logo.png",
    "JAMS Logo - 1.png",
    "JAMS Logo.png",
    "logo.png",
    "assets/jams_logo.png",
    "assets/logo.png",
    "static/jams_logo.png",
    "static/logo.png",
]


def find_logo_path() -> Optional[str]:
    cwd = os.getcwd()
    checks = []
    for c in LOGO_CANDIDATES:
        checks.append(os.path.join(cwd, c))

    mount_src = "/mount/src"
    if os.path.isdir(mount_src) and mount_src != cwd:
        for c in LOGO_CANDIDATES:
            checks.append(os.path.join(mount_src, c))
        try:
            for repo_name in os.listdir(mount_src):
                repo_path = os.path.join(mount_src, repo_name)
                if os.path.isdir(repo_path):
                    for c in LOGO_CANDIDATES:
                        checks.append(os.path.join(repo_path, c))
        except Exception:
            pass

    for p in checks:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


# =========================
# UI / THEME
# =========================
def inject_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --brand-red: {BRAND_RED};
                --bg: {BRAND_GRAY_BG};
                --border: {BRAND_GRAY_BORDER};
                --text: {TEXT_DARK};
            }}

            .stApp {{
                background: linear-gradient(180deg, var(--bg) 0%, #FFFFFF 60%);
                color: var(--text);
            }}

            section[data-testid="stSidebar"] {{
                background: #FFFFFF;
                border-right: 1px solid var(--border);
            }}

            h1, h2, h3 {{
                color: var(--brand-red);
                letter-spacing: -0.02em;
            }}

            .stButton > button {{
                background: var(--brand-red);
                color: white;
                border: 1px solid var(--brand-red);
                border-radius: 10px;
                padding: 0.55rem 0.9rem;
                font-weight: 700;
            }}
            .stButton > button:hover {{
                background: #7D1212;
                border-color: #7D1212;
            }}

            input:focus, textarea:focus {{
                border-color: var(--brand-red) !important;
                box-shadow: 0 0 0 0.12rem rgba(146, 21, 21, 0.18) !important;
            }}

            .jams-card {{
                background: #FFFFFF;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 22px 22px;
                box-shadow: 0 10px 22px rgba(0,0,0,0.04);
                min-height: 110px;
            }}

            .jams-muted {{
                color: #5B5B66;
                font-size: 13px;
            }}

            .jams-badge {{
                display: inline-block;
                background: rgba(146, 21, 21, 0.10);
                border: 1px solid rgba(146, 21, 21, 0.20);
                color: var(--brand-red);
                padding: 2px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 650;
                margin-left: 8px;
            }}

            .kpi-label {{
                font-size: 13px;
                color: #5B5B66;
                margin-bottom: 6px;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .kpi-value {{
                font-size: 38px;
                font-weight: 950;
                color: var(--brand-red);
                line-height: 1.05;
            }}
            .kpi-sub {{
                margin-top: 6px;
                font-size: 13px;
                color: #5B5B66;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HELPERS
# =========================
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_key(s: str) -> str:
    s = safe_str(s).lower().replace("\u00a0", " ")
    return " ".join(s.split())


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df.columns = [safe_str(c) for c in df.columns]
    return df.dropna(how="all")


def is_http_url(s: str) -> bool:
    s = safe_str(s).lower()
    return s.startswith("http://") or s.startswith("https://")


def parse_date_safe(s: str) -> Optional[date]:
    s = safe_str(s)
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def normalize_role(s: str) -> str:
    s = safe_str(s).lower()
    return "admin" if "admin" in s else "member"


def uniq_nonempty(series: pd.Series) -> int:
    if series is None:
        return 0
    vals = [safe_str(x) for x in series.dropna().tolist()]
    vals = [v for v in vals if v]
    return len(set(vals))


def extract_firm_from_experience(exp: str) -> str:
    exp = safe_str(exp)
    if not exp:
        return ""
    exp2 = exp.replace("—", "-").replace("–", "-").replace("|", "-").replace("@", "-").strip()
    parts = [p.strip() for p in exp2.split("-") if p.strip()]
    cand = parts[0] if parts else exp2
    cand = cand.split(",")[0].strip()
    cand = re.sub(r"\s*\(.*?\)\s*", "", cand).strip()
    if len(cand) < 3:
        return ""
    generic = {"intern", "analyst", "associate", "student", "research", "investment", "finance"}
    if cand.lower() in generic:
        return ""
    return cand


def count_firms(alumni_df: pd.DataFrame) -> int:
    if "Experience" not in alumni_df.columns:
        return 0
    firms = []
    for v in alumni_df["Experience"].tolist():
        f = extract_firm_from_experience(v)
        if f:
            firms.append(f)
    return len(set(firms))


def normalize_country_name(c: str) -> str:
    c = safe_str(c)
    if not c:
        return ""
    cl = c.lower()
    mapping = {
        "us": "United States",
        "u.s.": "United States",
        "usa": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    return mapping.get(cl, c)


# =========================
# KPI
# =========================
def render_kpi(container, label: str, value: int, sub: str = ""):
    container.markdown(
        f"""
        <div class="jams-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{int(value)}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def animate_kpi(container, label: str, target: int, sub: str = "", duration: float = 0.65):
    target = int(target or 0)
    ph = container.empty()

    if target <= 0:
        render_kpi(ph, label, 0, sub)
        return

    steps = 30
    sleep = duration / steps
    last_val = -1
    for i in range(steps):
        val = int((i + 1) * target / steps)
        if val == last_val:
            continue
        last_val = val
        render_kpi(ph, label, val, sub)
        time.sleep(sleep)


# =========================
# GOOGLE SHEETS LOADER
# =========================
def sheet_csv_export_url(sheet_id: str, tab_name: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={urllib.parse.quote(tab_name)}"
    )


def fetch_csv_text(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def detect_header_row(raw: pd.DataFrame, expected_headers: List[str]) -> Optional[int]:
    expected = {norm_key(h) for h in expected_headers}
    scan_max = min(len(raw), 80)
    for i in range(scan_max):
        row_vals = [norm_key(v) for v in raw.iloc[i].tolist()]
        found = {v for v in row_vals if v}
        if expected.issubset(found):
            return i
    return None


def build_df_from_raw(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    headers = [safe_str(x) for x in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    return clean_df(df).dropna(how="all")


@st.cache_data(ttl=120)
def load_sheet_tab(sheet_id: str, tab_name: str, expected_headers: List[str]) -> pd.DataFrame:
    url = sheet_csv_export_url(sheet_id, tab_name)
    text = fetch_csv_text(url)

    if text.lstrip().lower().startswith("<!doctype") or text.lstrip().lower().startswith("<html"):
        raise ValueError(f"Tab '{tab_name}': Google returned HTML instead of CSV (export blocked or tab mismatch).")

    if not text.strip():
        raise ValueError(f"Tab '{tab_name}': empty CSV response.")

    raw = pd.read_csv(io.StringIO(text), header=None)
    header_row = detect_header_row(raw, expected_headers)
    if header_row is None:
        sample = raw.head(8).fillna("").astype(str).values.tolist()
        raise ValueError(f"Tab '{tab_name}': Could not find expected headers. Sample rows: {sample}")

    return build_df_from_raw(raw, header_row)


# =========================
# DATABASE (SQLite)
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            name TEXT PRIMARY KEY,
            email TEXT,
            last_login_utc TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS career_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            service_type TEXT NOT NULL,
            requester_name TEXT NOT NULL,
            requester_email TEXT NOT NULL,
            message TEXT NOT NULL,
            alumni_english_name TEXT,
            alumni_chinese_name TEXT,
            alumni_email TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            decision_at TEXT,
            alumni_notes TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            title TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_time TEXT,
            location TEXT,
            link TEXT,
            description TEXT
        );
        """
    )

    conn.commit()
    conn.close()


def upsert_user_login(name: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (name, email, last_login_utc)
        VALUES (?, NULL, ?)
        ON CONFLICT(name) DO UPDATE SET last_login_utc = excluded.last_login_utc;
        """,
        (name, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_email(name: str) -> str:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE name = ?;", (name,))
    row = cur.fetchone()
    conn.close()
    return safe_str(row[0]) if row and row[0] else ""


def set_user_email(name: str, email: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET email = ? WHERE name = ?;", (email.strip().lower(), name))
    conn.commit()
    conn.close()


def create_request(service_type, requester_name, requester_email, message, alumni_english_name, alumni_chinese_name, alumni_email):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO career_requests
        (created_at, service_type, requester_name, requester_email, message,
         alumni_english_name, alumni_chinese_name, alumni_email, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending');
        """,
        (
            datetime.utcnow().isoformat(),
            service_type,
            requester_name,
            requester_email,
            message,
            alumni_english_name,
            alumni_chinese_name,
            alumni_email,
        ),
    )
    conn.commit()
    conn.close()


def get_requests_for_requester(requester_email: str) -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, service_type,
               alumni_english_name, alumni_chinese_name, alumni_email,
               status, decision_at, alumni_notes, message
        FROM career_requests
        WHERE requester_email = ?
        ORDER BY id DESC;
        """,
        conn,
        params=(requester_email,),
    )
    conn.close()
    return df


def get_requests_for_alumni(alumni_email: str) -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, service_type,
               requester_name, requester_email, message, status
        FROM career_requests
        WHERE alumni_email = ?
        ORDER BY id DESC;
        """,
        conn,
        params=(alumni_email,),
    )
    conn.close()
    return df


def set_request_status(req_id: int, status: str, alumni_notes: str = ""):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE career_requests
        SET status = ?, decision_at = ?, alumni_notes = ?
        WHERE id = ?;
        """,
        (status, datetime.utcnow().isoformat(), alumni_notes, req_id),
    )
    conn.commit()
    conn.close()


def create_event(created_by: str, title: str, event_date: str, event_time: str,
                 location: str, link: str, description: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events
        (created_at, created_by, title, event_date, event_time, location, link, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            datetime.utcnow().isoformat(),
            created_by,
            title.strip(),
            event_date.strip(),
            (event_time.strip() if event_time else ""),
            (location.strip() if location else ""),
            (link.strip() if link else ""),
            (description.strip() if description else ""),
        ),
    )
    conn.commit()
    conn.close()


def delete_event(event_id: int):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE id = ?;", (event_id,))
    conn.commit()
    conn.close()


def get_upcoming_events() -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query(
        """
        SELECT id, created_at, created_by, title, event_date, event_time, location, link, description
        FROM events
        ORDER BY event_date ASC, event_time ASC, id DESC;
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return df

    df["__dt"] = df["event_date"].apply(parse_date_safe)
    today = date.today()
    df = df[df["__dt"].notna()]
    df = df[df["__dt"] >= today].copy()
    return df.sort_values(by=["__dt", "event_time"], ascending=True)


# =========================
# AUTH
# =========================
def authenticate(passwords_df: pd.DataFrame, pw: str) -> Tuple[Optional[str], str]:
    pw = safe_str(pw)
    if not pw or passwords_df is None or passwords_df.empty:
        return None, ""
    match = passwords_df[passwords_df["Password"].astype(str) == pw]
    if match.empty:
        return None, ""
    name = safe_str(match.iloc[0]["Name"])
    role = normalize_role(match.iloc[0].get("Status", "member"))
    return name, role


def login_gate(passwords_df: pd.DataFrame, logo_path: Optional[str]) -> bool:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_role" not in st.session_state:
        st.session_state.user_role = "member"
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"

    if st.session_state.logged_in and st.session_state.user_name:
        return True

    if logo_path:
        st.image(logo_path, width=140)

    st.title("JAMS Investment Network")
    st.markdown("Enter your **individual password** to continue.")
    pw = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Sign in"):
        name, role = authenticate(passwords_df, pw)
        if not name:
            st.error("Incorrect password.")
            return False

        st.session_state.logged_in = True
        st.session_state.user_name = name
        st.session_state.user_role = role

        upsert_user_login(name)
        st.session_state.user_email = get_user_email(name)

        st.session_state.nav = "Home"
        st.rerun()

    return False


def compute_is_alumni(alumni_df: pd.DataFrame, email: str) -> bool:
    email = safe_str(email).lower()
    if not email or alumni_df is None or alumni_df.empty or "Email" not in alumni_df.columns:
        return False
    alumni_emails = set(alumni_df["Email"].astype(str).str.strip().str.lower())
    return email in alumni_emails


# =========================
# PLOTLY STYLING
# =========================
def fig_style(fig, title: str, height: int = 420):
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color=BRAND_RED, family="Arial"),
        font=dict(color=TEXT_DARK),
        legend_title_text="",
        height=height,
    )
    return fig


def get_location_col(alumni: pd.DataFrame) -> Optional[str]:
    for c in ["Country", "Location"]:
        if c in alumni.columns:
            return c
    return None


# =========================
# VISUALS
# =========================
def render_alumni_location_map(alumni: pd.DataFrame):
    loc_col = get_location_col(alumni)
    if not loc_col:
        st.info("Map unavailable: add a 'Location' (or 'Country') column in Alumni tab.")
        return

    tmp = alumni.copy()
    tmp[loc_col] = tmp[loc_col].apply(normalize_country_name)
    tmp = tmp[tmp[loc_col].astype(str).str.strip().ne("")]

    if tmp.empty:
        st.info("Map unavailable: no Location/Country values found.")
        return

    counts = (
        tmp.groupby(loc_col, as_index=False)
        .size()
        .rename(columns={loc_col: "Country", "size": "Alumni"})
        .sort_values("Alumni", ascending=False)
    )

    fig = px.choropleth(
        counts,
        locations="Country",
        locationmode="country names",
        color="Alumni",
        hover_name="Country",
        hover_data={"Alumni": True},
        projection="natural earth",
        color_continuous_scale=[
            [0.0, "#F6EDED"],
            [0.35, "#E3BDBD"],
            [0.70, "#B65B5B"],
            [1.0, BRAND_RED],
        ],
    )

    fig = fig_style(fig, "Alumni by Country", height=520)
    fig.update_coloraxes(showscale=False)

    # Make it "tech": dark ocean, transparent paper, no frame
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.20)",
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.18)",
        showland=True,
        landcolor="rgba(255,255,255,0.08)",
        showocean=True,
        oceancolor="rgba(12,12,16,0.94)",
        lakecolor="rgba(12,12,16,0.94)",
        showlakes=True,
        projection_scale=1.10,
    )

    # Stronger interactivity + nicer hover
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Alumni: %{z}<extra></extra>"
    )
    fig.update_layout(dragmode="pan")

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_top_firms_top3_plus_others(alumni: pd.DataFrame):
    if "Experience" not in alumni.columns:
        st.info("Top firms chart unavailable.")
        return

    firms = []
    for exp in alumni["Experience"].tolist():
        f = extract_firm_from_experience(exp)
        if f:
            firms.append(f)

    if not firms:
        st.info("Top firms chart unavailable: could not derive firms from Experience.")
        return

    vc = pd.Series(firms).value_counts()
    top3 = vc.head(3)
    others = int(vc.iloc[3:].sum())

    data = pd.DataFrame({"Firm": list(top3.index), "Alumni": list(top3.values)})
    if others > 0:
        data = pd.concat([data, pd.DataFrame([{"Firm": "Others", "Alumni": others}])], ignore_index=True)

    data = data.sort_values("Alumni", ascending=True)

    fig = px.bar(data, x="Alumni", y="Firm", orientation="h")
    fig.update_traces(marker_color=BRAND_RED)
    fig = fig_style(fig, "Top Firms (Top 3 + Others)", height=420)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_major_top5(alumni: pd.DataFrame):
    if "Major" not in alumni.columns:
        st.info("Major chart unavailable.")
        return

    tmp = alumni.copy()
    tmp["Major"] = tmp["Major"].astype(str).str.strip()
    tmp = tmp[tmp["Major"].ne("")]

    if tmp.empty:
        st.info("Major chart unavailable: no Major values.")
        return

    vc = tmp["Major"].value_counts()
    top5 = vc.head(5)
    others = int(vc.iloc[5:].sum())

    data = pd.DataFrame({"Major": list(top5.index), "Alumni": list(top5.values)})
    if others > 0:
        data = pd.concat([data, pd.DataFrame([{"Major": "Others", "Alumni": others}])], ignore_index=True)

    # Explicit JAMS palette (ensures donut is #921515 themed)
    fig = px.pie(
        data,
        names="Major",
        values="Alumni",
        hole=0.60,
        color="Major",
        color_discrete_sequence=JAMS_RED_SCALE,
    )

    # Make "Others" a neutral gray so red reads as primary (optional)
    if "Others" in data["Major"].tolist():
        fig.update_traces(
            marker=dict(
                colors=[
                    (JAMS_RED_SCALE[i] if name != "Others" else "#B9BAC3")
                    for i, name in enumerate(data["Major"].tolist())
                ]
            )
        )

    fig = fig_style(fig, "Alumni by Major (Top 5 + Others)", height=420)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# =========================
# PAGES
# =========================
def page_home(alumni: pd.DataFrame, resources: pd.DataFrame, logo_path: Optional[str]):
    h1, h2 = st.columns([1, 7], vertical_alignment="center")
    with h1:
        if logo_path:
            st.image(logo_path, width=120)
        else:
            st.markdown(
                "<div class='jams-muted'>Logo not found in repo.<br/>Add <b>jams_logo.png</b> next to app.py.</div>",
                unsafe_allow_html=True,
            )

    with h2:
        st.markdown(
            f"""
            <div style="font-size:36px; font-weight:950; color:{BRAND_RED}; line-height:1.1;">
              JAMS Investment Network Platform
            </div>
            <div class="jams-muted" style="margin-top:10px; max-width:1200px; font-size:15px; line-height:1.55;">
              {HOME_INTRO}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    alumni_count = len(alumni) if alumni is not None else 0
    schools_count = uniq_nonempty(alumni["School"]) if "School" in alumni.columns else 0
    majors_count = uniq_nonempty(alumni["Major"]) if "Major" in alumni.columns else 0
    firms_count = count_firms(alumni)
    resources_count = len(resources) if resources is not None else 0
    events_upcoming = len(get_upcoming_events())

    if "kpi_animated" not in st.session_state:
        st.session_state.kpi_animated = False

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    if st.session_state.kpi_animated:
        render_kpi(r1c1, "Alumni", alumni_count, "Total in directory")
        render_kpi(r1c2, "Schools", schools_count, "Unique schools represented")
        render_kpi(r1c3, "Majors", majors_count, "Unique majors represented")
        render_kpi(r2c1, "Firms", firms_count, "Derived from Experience")
        render_kpi(r2c2, "Resources", resources_count, "Shared materials")
        render_kpi(r2c3, "Events", events_upcoming, "Upcoming published")
    else:
        animate_kpi(r1c1, "Alumni", alumni_count, "Total in directory")
        animate_kpi(r1c2, "Schools", schools_count, "Unique schools represented")
        animate_kpi(r1c3, "Majors", majors_count, "Unique majors represented")
        animate_kpi(r2c1, "Firms", firms_count, "Derived from Experience")
        animate_kpi(r2c2, "Resources", resources_count, "Shared materials")
        animate_kpi(r2c3, "Events", events_upcoming, "Upcoming published")
        st.session_state.kpi_animated = True

    st.write("")
    st.markdown("## Network Visualizations")

    render_alumni_location_map(alumni)

    c1, c2 = st.columns([1, 1])
    with c1:
        render_top_firms_top3_plus_others(alumni)
    with c2:
        render_major_top5(alumni)


def page_alumni(alumni: pd.DataFrame):
    st.header("Alumni Directory")
    missing = [c for c in ALUMNI_HEADERS if c not in alumni.columns]
    if missing:
        st.error(f"Alumni tab missing columns: {missing}")
        st.caption(f"Detected columns: {list(alumni.columns)}")
        return

    f1, f2, f3 = st.columns([2, 2, 4])
    with f1:
        cohort_vals = sorted({safe_str(x) for x in alumni["Cohort Year"].dropna().unique() if safe_str(x)})
        cohort = st.selectbox("Filter: Cohort Year", ["All"] + cohort_vals)
    with f2:
        major_vals = sorted({safe_str(x) for x in alumni["Major"].dropna().unique() if safe_str(x)})
        major = st.selectbox("Filter: Major", ["All"] + major_vals)
    with f3:
        q = st.text_input("Search", placeholder="name, experience, school, department, etc.")

    df = alumni.copy()
    if cohort != "All":
        df = df[df["Cohort Year"].astype(str).str.strip() == cohort]
    if major != "All":
        df = df[df["Major"].astype(str).str.strip() == major]
    if q:
        ql = q.lower().strip()
        df = df[df.apply(lambda r: any(ql in safe_str(v).lower() for v in r.values), axis=1)]

    st.caption(f"Showing {len(df)} alumni")

    for _, r in df.iterrows():
        en = safe_str(r["English Name"])
        zh = safe_str(r["Chinese Name"])
        title = en or zh or "Alumni"

        exp = safe_str(r["Experience"])
        dept = safe_str(r["JAMS Department"])
        school = safe_str(r["School"])
        major_v = safe_str(r["Major"])
        cohort_v = safe_str(r["Cohort Year"])
        email = safe_str(r["Email"])
        linkedin = safe_str(r["Linkedin"])
        loc = safe_str(r.get("Location", ""))

        subtitle_line = " • ".join(
            [p for p in [zh, dept, school, major_v, loc, (f"Cohort {cohort_v}" if cohort_v else "")] if p]
        )

        with st.container(border=True):
            st.markdown(f"### {title}")
            if subtitle_line:
                st.caption(subtitle_line)
            if exp:
                st.write(exp)

            link_cols = st.columns(2)
            with link_cols[0]:
                if linkedin and is_http_url(linkedin):
                    st.link_button("LinkedIn", linkedin)
            with link_cols[1]:
                if email:
                    mailto = f"mailto:{email}?{urllib.parse.urlencode({'subject':'JAMS Network','body':f'Hi {title},\\n\\nI’m a JAMS member and would love to connect.\\n\\nBest,\\n{st.session_state.user_name}\\n'})}"
                    st.link_button("Email", mailto)


def page_resources(resources: pd.DataFrame):
    st.header("Resources")
    missing = [c for c in RES_HEADERS if c not in resources.columns]
    if missing:
        st.error(f"Resources tab missing columns: {missing}")
        st.caption(f"Detected columns: {list(resources.columns)}")
        return

    q = st.text_input("Search resources", placeholder="keywords in title/description")
    df = resources.copy()
    if q:
        ql = q.lower().strip()
        df = df[df.apply(lambda r: any(ql in safe_str(v).lower() for v in r.values), axis=1)]

    if df.empty:
        st.info("No resources found.")
        return

    for i, r in df.iterrows():
        title = safe_str(r["Title"]) or "Resource"
        desc = safe_str(r["Description"])
        link = safe_str(r["Link"])

        with st.container(border=True):
            st.markdown(f"### {title}")
            if desc:
                st.write(desc)
            if link and is_http_url(link):
                st.link_button("Open Resource", link)
            elif link:
                st.text_input("Resource link/file", value=link, disabled=True, key=f"res_{i}")


def page_career_support(alumni: pd.DataFrame):
    st.header("Career Support")
    st.caption("3 steps: select service → select alumni → submit request (stored in-platform).")

    if not st.session_state.user_email:
        st.warning("Set your email in the sidebar first (one-time).")
        return

    df = alumni.copy()
    df = df[df["Email"].astype(str).str.strip().ne("")]

    service = st.selectbox("Step 1 — Career service", CAREER_SERVICES)

    options: List[Tuple[str, str, str, str]] = []
    for _, r in df.iterrows():
        en = safe_str(r["English Name"])
        zh = safe_str(r["Chinese Name"])
        em = safe_str(r["Email"]).lower()
        cohort_year = safe_str(r["Cohort Year"])
        label = f"{en} ({zh}) • Cohort {cohort_year}" if zh else f"{en} • Cohort {cohort_year}"
        options.append((label, en, zh, em))

    pick = st.selectbox("Step 2 — Choose alumni", [o[0] for o in options])
    _, alumni_en, alumni_zh, alumni_email = next(o for o in options if o[0] == pick)
    message = st.text_area("Step 3 — Message", height=160)

    if st.button("Submit request"):
        if not message.strip():
            st.error("Please include a message.")
            return
        create_request(
            service_type=service,
            requester_name=st.session_state.user_name,
            requester_email=st.session_state.user_email,
            message=message.strip(),
            alumni_english_name=alumni_en,
            alumni_chinese_name=alumni_zh,
            alumni_email=alumni_email,
        )
        st.success("Request submitted.")


def page_my_requests():
    st.header("My Career Requests")
    if not st.session_state.user_email:
        st.warning("Set your email in the sidebar first (one-time).")
        return
    df = get_requests_for_requester(st.session_state.user_email)
    if df.empty:
        st.info("No requests found for your email.")
        return
    for _, r in df.iterrows():
        with st.container(border=True):
            st.markdown(f"### #{int(r['id'])} — {safe_str(r['service_type'])}")
            st.caption(f"Status: {safe_str(r['status'])}")
            st.write(safe_str(r["message"]))


def page_alumni_approvals():
    st.header("Alumni Approvals")
    df = get_requests_for_alumni(st.session_state.user_email)
    if df.empty:
        st.info("No requests assigned to your email.")
        return
    for _, r in df.iterrows():
        req_id = int(r["id"])
        with st.container(border=True):
            st.markdown(f"### #{req_id} — {safe_str(r['service_type'])}")
            st.caption(f"From: {safe_str(r.get('requester_name',''))} ({safe_str(r['requester_email'])})")
            st.write(safe_str(r["message"]))
            if safe_str(r["status"]).lower() == "pending":
                notes = st.text_input("Notes (optional)", key=f"notes_{req_id}")
                a, d = st.columns(2)
                if a.button("Approve", key=f"approve_{req_id}"):
                    set_request_status(req_id, "Approved", notes)
                    st.rerun()
                if d.button("Decline", key=f"decline_{req_id}"):
                    set_request_status(req_id, "Declined", notes)
                    st.rerun()


def page_events_member_view():
    st.header("Upcoming Events")
    df = get_upcoming_events()
    if df.empty:
        st.info("No upcoming events published yet.")
        return
    for _, r in df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {safe_str(r['title'])}")
            st.caption(" • ".join([p for p in [safe_str(r["event_date"]), safe_str(r["event_time"]), safe_str(r["location"])] if p]))
            if safe_str(r["description"]):
                st.write(safe_str(r["description"]))
            if safe_str(r["link"]) and is_http_url(safe_str(r["link"])):
                st.link_button("Open Link", safe_str(r["link"]))


def page_events_admin():
    st.header("Events Admin")
    with st.expander("Publish a new event", expanded=True):
        title = st.text_input("Title")
        event_date = st.text_input("Date (YYYY-MM-DD)", placeholder="2026-01-15")
        event_time = st.text_input("Time", placeholder="7:00 PM ET")
        location = st.text_input("Location", placeholder="Zoom / Campus / Address")
        link = st.text_input("Link (optional)", placeholder="https://...")
        description = st.text_area("Description", height=120)
        if st.button("Publish event"):
            if not title.strip():
                st.error("Title is required.")
                return
            if not event_date.strip() or parse_date_safe(event_date.strip()) is None:
                st.error("Valid Date is required (YYYY-MM-DD).")
                return
            create_event(st.session_state.user_name, title, event_date, event_time, location, link, description)
            st.success("Event published.")
            st.rerun()

    st.divider()
    df = get_upcoming_events()
    if df.empty:
        st.info("No upcoming events.")
        return
    for _, r in df.iterrows():
        event_id = int(r["id"])
        with st.container(border=True):
            st.markdown(f"### {safe_str(r['title'])}")
            st.caption(" • ".join([p for p in [safe_str(r["event_date"]), safe_str(r["event_time"]), safe_str(r["location"])] if p]))
            if st.button("Delete", key=f"del_event_{event_id}"):
                delete_event(event_id)
                st.rerun()


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="JAMS Investment Network", layout="wide")
    inject_css()
    init_db()

    logo_path = find_logo_path()

    passwords = load_sheet_tab(SHEET_ID, TAB_PASSWORDS, PWD_HEADERS)
    passwords = clean_df(passwords)

    if not login_gate(passwords, logo_path):
        return

    try:
        alumni = load_sheet_tab(SHEET_ID, TAB_ALUMNI, ALUMNI_HEADERS)
        resources = load_sheet_tab(SHEET_ID, TAB_RESOURCES, RES_HEADERS)
    except Exception as e:
        st.error("Could not load Google Sheets data.")
        st.code(str(e))
        st.caption("If you recently changed headers, click 'Clear cache' (menu) or wait 2 minutes (cache TTL).")
        return

    st.sidebar.title("JAMS Platform")
    role_badge = "Admin" if st.session_state.user_role == "admin" else "Member"
    st.sidebar.markdown(
        f"**Welcome back, {st.session_state.user_name}** <span class='jams-badge'>{role_badge}</span>",
        unsafe_allow_html=True,
    )
    if logo_path:
        st.sidebar.image(logo_path, width=120)

    st.sidebar.markdown("### Your email (one-time)")
    current_email = st.session_state.user_email or ""
    new_email = st.sidebar.text_input("Email", value=current_email, placeholder="name@email.com").strip().lower()
    if new_email and new_email != current_email:
        set_user_email(st.session_state.user_name, new_email)
        st.session_state.user_email = new_email

    st.session_state.is_alumni = compute_is_alumni(alumni, st.session_state.user_email)

    pages = ["Home", "Alumni Directory", "Resources", "Events", "Career Support", "My Career Requests"]
    if st.session_state.is_alumni:
        pages.append("Alumni Approvals")
    if st.session_state.user_role == "admin":
        pages.append("Events Admin")

    if "nav" not in st.session_state or st.session_state.nav not in pages:
        st.session_state.nav = "Home"

    st.sidebar.radio("Navigate", pages, key="nav")
    page = st.session_state.nav

    if page == "Home":
        page_home(alumni, resources, logo_path)
    elif page == "Alumni Directory":
        page_alumni(alumni)
    elif page == "Resources":
        page_resources(resources)
    elif page == "Events":
        page_events_member_view()
    elif page == "Career Support":
        page_career_support(alumni)
    elif page == "My Career Requests":
        page_my_requests()
    elif page == "Alumni Approvals":
        page_alumni_approvals()
    elif page == "Events Admin":
        page_events_admin()


if __name__ == "__main__":
    main()
