# app.py  ─ Streamlit (폰트 견고화 / 캐시초기화 / 최신 파일 자동선택 / TTL 캐시 / CSV다운로드
#                    / 표 숫자 중앙정렬 / 그래프 라벨(현재·계획·부족·초과) / "연도별 → 요약표" 순서로 배치
#                    / [신규] 최상단: 현재연도 입주 실적/계획 요약 → 지도(입주율 ↔ 계획대비 편차) → 기존 내용)
import os, logging, warnings, shutil, time, hashlib, io, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# 동적 그래프와 서브플롯을 위한 Plotly 임포트
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(page_title="입주율 분석", layout="wide")

# -------------------- 코드 버전(파일 해시) --------------------
def _code_digest() -> str:
    try:
        p = Path(__file__)
        return hashlib.md5(p.read_bytes()).hexdigest()[:10]
    except Exception:
        return time.strftime("ts%Y%m%d%H%M%S", time.localtime())

CODE_VER = _code_digest()

# -------------------- 한글 폰트 적용(강력) --------------------
def set_korean_font_strict():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    local_candidates = [
        os.path.abspath("fonts/NanumGothic-Regular.ttf"),
        os.path.abspath("fonts/NotoSansKR-Regular.otf"),
        "assets/fonts/NanumGothic.ttf",
        "assets/fonts/NotoSansKR-Regular.otf",
    ]
    system_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    candidates = local_candidates + system_candidates
    chosen_path = None
    for p in candidates:
        if p and os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                chosen_path = p
                break
            except Exception:
                continue
    if chosen_path:
        prop = fm.FontProperties(fname=chosen_path)
        chosen_name = prop.get_name()
        try:
            cache_dir = fm.get_cachedir()
            if cache_dir and os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        mpl.rcParams["font.family"] = [chosen_name, "DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = [chosen_name, "DejaVu Sans"]
    else:
        chosen_name = "DejaVu Sans"
        mpl.rcParams["font.family"] = [chosen_name]
    mpl.rcParams["axes.unicode_minus"] = False
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    return chosen_name

def apply_korean_font(fig):
    fam = mpl.rcParams.get("font.family", ["DejaVu Sans"])
    fam = fam[0] if isinstance(fam, (list, tuple)) else fam
    kprop = fm.FontProperties(family=fam)
    for ax in fig.get_axes():
        if ax.title:
            ax.title.set_fontproperties(kprop)
        if ax.xaxis and ax.xaxis.get_label():
            ax.xaxis.get_label().set_fontproperties(kprop)
        if ax.yaxis and ax.yaxis.get_label():
            ax.yaxis.get_label().set_fontproperties(kprop)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(kprop)
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontproperties(kprop)
        for child in ax.get_children():
            if hasattr(child, "get_celld"):
                for cell in child.get_celld().values():
                    cell._text.set_fontproperties(kprop)

chosen_font = set_korean_font_strict()

# -------------------- 표 중앙정렬(CSS) --------------------
def inject_centered_style():
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] div[role="gridcell"]{display:flex;justify-content:center !important;}
        [data-testid="stDataFrame"] div[role="columnheader"]{display:flex;justify-content:center !important;}
        [data-testid="stDataFrame"] table td,[data-testid="stDataFrame"] table th{ text-align:center !important;}
        [data-testid="stDataFrame"] table td div,[data-testid="stDataFrame"] table th div{justify-content:center !important;}
        [data-testid="stDataFrame"] thead tr th div[role="button"]{justify-content:center !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_centered_style()

# -------------------- 사이드바 --------------------
try:
    st.sidebar.image("logo.png", use_container_width=True)
except Exception:
    pass

st.sidebar.markdown("**🏢 마케팅본부 마케팅팀**")
st.sidebar.divider()

top_container = st.sidebar.container()
st.sidebar.divider()

st.sidebar.markdown("### 데이터 / 필터")
load_way = st.sidebar.radio("데이터 불러오기 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

auto_pick_latest = st.sidebar.checkbox("최신 파일 자동 선택(패턴)", value=True)
pattern = st.sidebar.text_input("패턴(자동 선택)", value="입주율*.xlsx")

uploaded_file = None
if load_way == "Repo 내 파일 사용":
    excel_path = st.sidebar.text_input("엑셀 파일 경로(수동)", value="입주율.xlsx")
else:
    uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=["xlsx"])
    excel_path = None

if st.sidebar.button("데이터 캐시 초기화"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("캐시 초기화 완료")
    st.rerun()

ttl_minutes = st.sidebar.number_input("자동 갱신 주기(TTL, 분)", min_value=0, max_value=120, value=0, step=5)

# -------------------- 데이터 로드 --------------------
def file_digest_from_path(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:10]

def file_digest_from_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()[:10]

def ttl_bucket(minutes: int) -> str:
    if not minutes or minutes <= 0:
        return "ttl0"
    return f"ttl{int(time.time() // (minutes * 60))}"

COORD_COLS = ["위도", "경도"]

@st.cache_data(show_spinner=False)
def _has_coords(path_str: str, mtime: float) -> bool:
    """엑셀 'data' 시트에 위도/경도 컬럼이 있는지 헤더만 읽어서 확인"""
    try:
        cols = pd.read_excel(path_str, sheet_name="data", nrows=0).columns
        return all(c in cols for c in COORD_COLS)
    except Exception:
        return False

def _finalize_df(df_local: pd.DataFrame, location_df: pd.DataFrame | None) -> pd.DataFrame:
    df_local["공급승인일자"] = pd.to_datetime(df_local["공급승인일자"], errors="coerce")
    # 위도/경도가 data 시트에 없으면 '위치' 시트에서 아파트코드로 붙여줌
    if not all(c in df_local.columns for c in COORD_COLS) and location_df is not None:
        if {"아파트코드", *COORD_COLS}.issubset(location_df.columns) and "아파트코드" in df_local.columns:
            loc = location_df[["아파트코드", *COORD_COLS]].dropna(subset=COORD_COLS).drop_duplicates("아파트코드")
            df_local = df_local.drop(columns=[c for c in COORD_COLS if c in df_local.columns]).merge(
                loc, on="아파트코드", how="left"
            )
    for c in COORD_COLS:
        if c not in df_local.columns:
            df_local[c] = np.nan
        df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
    return df_local

def _read_location_sheet(src):
    try:
        return pd.read_excel(src, sheet_name="위치")
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_df_from_path_or_buffer(path_str: str | None, buffer_bytes: bytes | None, digest: str, ttl_key: str):
    if buffer_bytes is not None:
        df_local = pd.read_excel(io.BytesIO(buffer_bytes), sheet_name="data")
        loc_df = _read_location_sheet(io.BytesIO(buffer_bytes))
    else:
        if not path_str or not os.path.exists(path_str):
            return pd.DataFrame()
        df_local = pd.read_excel(path_str, sheet_name="data")
        loc_df = _read_location_sheet(path_str)
    return _finalize_df(df_local, loc_df)

selected_path_str = None
auto_hint = ""

def _all_xlsx_candidates() -> list:
    """패턴에 맞는 파일 + 폴더 내 모든 .xlsx (임시파일 ~$ 제외)"""
    seen, out = set(), []
    for pat in [pattern, "*.xlsx", "**/*.xlsx"]:
        for f in glob.glob(pat, recursive=True):
            fp = Path(f)
            if fp.name.startswith("~$") or str(fp) in seen:
                continue
            seen.add(str(fp)); out.append(fp)
    return out

if load_way == "Repo 내 파일 사용":
    if auto_pick_latest:
        matches = sorted(glob.glob(pattern))
        coord_files = [fp for fp in _all_xlsx_candidates() if _has_coords(str(fp), fp.stat().st_mtime)]
        if matches or coord_files:
            # 위도/경도가 포함된 파일을 우선, 그 다음 최신 수정시간 순
            pool = coord_files if coord_files else [Path(m) for m in matches]
            paths = sorted(pool, key=lambda p: p.stat().st_mtime, reverse=True)
            selected_path_str = str(paths[0])
            auto_hint = f"(자동선택: {Path(selected_path_str).name} · 좌표포함 파일 우선)"
        else:
            selected_path_str = excel_path
            auto_hint = "(패턴 일치 없음 → 수동 경로 사용)"
    else:
        selected_path_str = excel_path

data_caption = ""
ttl_key = ttl_bucket(ttl_minutes)

def _attach_coords_from_repo(_df: pd.DataFrame, exclude: str | None) -> pd.DataFrame:
    """선택된 파일에 좌표가 없으면, 폴더 내 다른 엑셀(좌표 포함)에서 아파트코드로 위도/경도를 붙임"""
    if _df.empty or _df[COORD_COLS].notna().any().any():
        return _df
    for fp in sorted(_all_xlsx_candidates(), key=lambda q: q.stat().st_mtime, reverse=True):
        if exclude and str(fp) == exclude:
            continue
        if not _has_coords(str(fp), fp.stat().st_mtime):
            continue
        other = load_df_from_path_or_buffer(str(fp), None, file_digest_from_path(fp), ttl_key)
        loc = other[["아파트코드", *COORD_COLS]].dropna(subset=COORD_COLS).drop_duplicates("아파트코드")
        merged = _df.drop(columns=COORD_COLS).merge(loc, on="아파트코드", how="left")
        return merged
    return _df

if load_way == "Repo 내 파일 사용":
    p = Path(selected_path_str) if selected_path_str else None
    if p and p.exists():
        digest = file_digest_from_path(p)
        df = load_df_from_path_or_buffer(str(p), None, digest, ttl_key)
        df = _attach_coords_from_repo(df, str(p))
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        n_coord = int(df[COORD_COLS].notna().all(axis=1).sum()) if not df.empty else 0
        data_caption = (f"📄 Source=Repo:{p.name} {auto_hint} | ver={digest} | updated={mtime} | {ttl_key}"
                        f" | 좌표 {n_coord}/{len(df)}단지")
    else:
        df = pd.DataFrame()
        data_caption = f"📄 Source=Repo: (경로 없음) | {ttl_key}"
else:
    if uploaded_file:
        b = uploaded_file.getvalue()
        digest = file_digest_from_bytes(b)
        df = load_df_from_path_or_buffer(None, b, digest, ttl_key)
        data_caption = f"📄 Source=Upload:{uploaded_file.name} | ver={digest} | {ttl_key}"
    else:
        df = pd.DataFrame()
        data_caption = f"📄 Source=Upload: (파일 미선택) | {ttl_key}"

# -------------------- 공통 유틸 --------------------
def ensure_start_index(_df: pd.DataFrame):
    month_cols = [c for c in _df.columns if "개월" in str(c)]

    def _key(c):
        s = "".join(ch for ch in str(c) if ch.isdigit())
        return int(s) if s else 0

    month_cols = sorted(month_cols, key=_key)

    def first_valid_idx(row):
        for i, col in enumerate(month_cols):
            v = row.get(col, np.nan)
            if pd.notna(v) and v > 0:
                return i
        return np.nan

    if "입주시작index" not in _df.columns:
        _df["입주시작index"] = _df.apply(first_valid_idx, axis=1)

    _df["입주시작월"] = _df.apply(
        lambda r: r["공급승인일자"] + pd.DateOffset(months=int(r["입주시작index"]))
        if pd.notna(r["입주시작index"]) and pd.notna(r["공급승인일자"])
        else pd.NaT,
        axis=1,
    )
    return month_cols

def _bubble_area_from_units(units, min_area=250, max_area=2800):
    v = pd.Series(units).fillna(0).astype(float).to_numpy()
    v = np.clip(v, 0, None)
    r = np.sqrt(v)
    r_min, r_max = r.min(), r.max()
    if r_max - r_min < 1e-9:
        return np.full_like(r, (min_area + max_area) / 2.0)
    return min_area + (r - r_min) / (r_max - r_min) * (max_area - min_area)

def _safe_ratio(num, den):
    if den and den > 0 and pd.notna(num):
        return float(np.clip(num / den, 0.0, 1.0))
    return np.nan

def _fmt_date_str(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

def _format_pct_cols(df_in, cols):
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")
    return df

# -------------------- [신규] 상단 요약/지도용 공통 헬퍼 (기존 함수는 수정하지 않음) --------------------
PLAN_PCT_TOP = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
_PLAN_TOP = {k: min(1.0, v / 100) for k, v in PLAN_PCT_TOP.items()}

def _plan_ratio_hold(m):
    """n개월차 계획 누적 입주율(0~1). 계획표는 1~9개월만 존재.
    10~12개월: 9개월 계획값 유지 / 13개월 이상: 100% (※ 상단 요약·지도 전용)"""
    m = int(m)
    if m in _PLAN_TOP:
        return _PLAN_TOP[m]
    if 9 < m <= 12:
        return _PLAN_TOP[9]
    if m > 12:
        return 1.0
    return np.nan

def _cum_rate_top(row, m, month_cols):
    idx = int(row["입주시작index"])
    cols = month_cols[idx: idx + m]
    num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols])
    return _safe_ratio(num, row["세대수"])

def _months_elapsed(row, ref_date):
    if pd.isna(row.get("입주시작월")):
        return 0
    delta = (ref_date.year - row["입주시작월"].year) * 12 + (ref_date.month - row["입주시작월"].month) + 1
    return max(0, delta)

def _map_zoom(lats, lons):
    span = max(float(np.ptp(lons)), float(np.ptp(lats)) * 1.6, 0.005)
    return float(np.clip(np.log2(560 / span), 8, 16))

def render_main_map(options, key, height=560):
    """options: {라벨: dict(df=, col=, mode='rate'|'diff', hover=[(라벨, 컬럼, 종류)])}
    좌측 상단 라디오 버튼으로 색상/대상 전환. 점 크기=세대수."""
    labels = list(options.keys())
    c1, c2 = st.columns([3, 1])
    sel = c1.radio("🎨 지도 보기", labels, horizontal=True, key=f"{key}_sel") if len(labels) > 1 else labels[0]
    show_names = c2.checkbox("단지명 표시", value=False, key=f"{key}_names")

    opt = options[sel]
    m_all = opt["df"]
    if m_all is None or m_all.empty:
        st.info("🗺️ 지도에 표시할 단지가 없어.")
        return
    if m_all[COORD_COLS].notna().sum().min() == 0:
        st.info("🗺️ 위도/경도 데이터가 없어서 지도를 표시할 수 없어. (좌표가 포함된 엑셀을 사용해 줘)")
        return
    m = m_all.dropna(subset=COORD_COLS + [opt["col"]]).copy()
    n_missing = len(m_all) - len(m)
    if m.empty:
        st.info("🗺️ 지도에 표시할 단지가 없어.")
        return
    vals = pd.to_numeric(m[opt["col"]], errors="coerce")

    if opt["mode"] == "rate":
        cmin, cmax, cbar = 0.0, 1.0, dict(title=sel, tickformat=".0%")
    else:
        lim = max(float(np.nanmax(np.abs(vals))), 1.0)
        cmin, cmax, cbar = -lim, lim, dict(title=sel, ticksuffix="pp")

    units = pd.to_numeric(m["세대수"], errors="coerce").fillna(0).clip(lower=0)
    r = np.sqrt(units.to_numpy())
    sizes = np.full(len(m), 16.0) if r.max() - r.min() < 1e-9 else 9 + (r - r.min()) / (r.max() - r.min()) * 21

    def _fmt(v, kind):
        if pd.isna(v):
            return "-"
        if kind == "int":
            return f"{int(round(v)):,}"
        if kind == "pct":
            return f"{v*100:.1f}%"
        if kind == "pp":
            return f"{v:+.1f}pp"
        if kind == "date":
            return pd.to_datetime(v).strftime("%Y-%m-%d")
        return str(v)

    hover = []
    for _, row in m.iterrows():
        lines = [f"<b>{row['아파트명']}</b>", f"세대수: {_fmt(row['세대수'], 'int')}"]
        for lab, col, kind in opt.get("hover", []):
            if col in m.columns:
                lines.append(f"{lab}: {_fmt(row[col], kind)}")
        hover.append("<br>".join(lines))

    common = dict(
        lat=m["위도"], lon=m["경도"], mode="markers+text" if show_names else "markers",
        marker=dict(size=sizes, color=vals, colorscale="RdYlGn", cmin=cmin, cmax=cmax, colorbar=cbar, opacity=0.85),
        text=m["아파트명"] if show_names else None, textposition="top center", textfont=dict(size=10),
        hovertext=hover, hoverinfo="text", name="",
    )
    center = dict(lat=float(m["위도"].mean()), lon=float(m["경도"].mean()))
    zoom = _map_zoom(m["위도"], m["경도"])
    fig = go.Figure()
    if hasattr(go, "Scattermap"):  # plotly >= 5.24
        fig.add_trace(go.Scattermap(**common))
        fig.update_layout(map=dict(style="open-street-map", center=center, zoom=zoom))
    else:
        fig.add_trace(go.Scattermapbox(**common))
        fig.update_layout(mapbox=dict(style="open-street-map", center=center, zoom=zoom))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height)
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart")
    cap = f"🔵 원 크기 = 세대수 · 색 = {sel} · 표시 {len(m)}개 단지"
    if n_missing:
        cap += f" · 좌표/값 없음 {n_missing}개 제외"
    st.caption(cap)

def build_top_map_options(시작일, 종료일, min_units):
    """지도용 데이터: (1) 분석기간 입주시작 단지의 입주율 (2) 2025년 이후 입주시작 단지의 계획 대비 편차"""
    month_cols = ensure_start_index(df)
    ok = df["입주시작index"].notna() & df["세대수"].notna() & (df["세대수"] >= min_units)

    # (1) 입주율 — '입주현황 요약표'와 동일 기준(분석기간 내 입주시작, 종료일까지 누적)
    a = df[ok & (df["입주시작월"] >= 시작일) & (df["입주시작월"] <= 종료일)].copy()
    if not a.empty:
        a["입주세대수"] = a.apply(
            lambda r: sum(0 if pd.isna(r.get(c)) else r.get(c)
                          for c in month_cols[int(r["입주시작index"]): int(r["입주시작index"]) + max(1, _months_elapsed(r, 종료일))]),
            axis=1)
        a["입주율"] = a.apply(lambda r: _cum_rate_top(r, max(1, _months_elapsed(r, 종료일)), month_cols), axis=1)
        a["잔여세대수"] = (a["세대수"] - a["입주세대수"]).clip(lower=0)

    # (2) 계획 대비 편차 — '계획 대비 저조/우수' 섹션과 동일 기준(2025-01 이후 입주시작, 종료일 기준)
    b = df[ok & (df["입주시작월"] >= pd.Timestamp("2025-01-01")) & (df["입주시작월"] <= 종료일)].copy()
    if not b.empty:
        b["경과개월"] = b.apply(lambda r: _months_elapsed(r, 종료일), axis=1)
        b = b[b["경과개월"] > 0].copy()
        b["실제누적(비율)"] = b.apply(lambda r: _cum_rate_top(r, int(r["경과개월"]), month_cols), axis=1)
        b["계획누적(비율)"] = b["경과개월"].apply(_plan_ratio_hold)
        b["실제누적세대"] = (b["실제누적(비율)"] * b["세대수"]).round()
        b["계획누적세대"] = (b["계획누적(비율)"] * b["세대수"]).round()
        b["부족세대"] = (b["계획누적세대"] - b["실제누적세대"]).clip(lower=0)
        b["편차(pp)"] = (b["실제누적(비율)"] - b["계획누적(비율)"]) * 100

    return {
        "입주율": dict(df=a, col="입주율", mode="rate", hover=[
            ("입주시작월", "입주시작월", "date"), ("입주세대수", "입주세대수", "int"),
            ("잔여세대수", "잔여세대수", "int"), ("입주율", "입주율", "pct")]),
        "계획 대비 편차": dict(df=b, col="편차(pp)", mode="diff", hover=[
            ("입주시작월", "입주시작월", "date"), ("경과개월", "경과개월", "int"),
            ("실제누적세대", "실제누적세대", "int"), ("계획누적세대", "계획누적세대", "int"),
            ("부족세대", "부족세대", "int"), ("실제누적", "실제누적(비율)", "pct"),
            ("계획누적", "계획누적(비율)", "pct"), ("편차", "편차(pp)", "pp")]),
    }

# -------------------- 종료일 디폴트: 엑셀 내 가장 최신 날짜 찾기 --------------------
def _last_data_date_from_df(_df: pd.DataFrame) -> pd.Timestamp | None:
    if _df is None or _df.empty:
        return None

    priority_keywords = ["데이터", "기준", "마감", "컷오프", "cutoff", "집계", "최종", "마지막"]
    # 🛠 [버그 수정] '공급승인일자'는 분양(공급) 승인일일 뿐 실제 입주 데이터가 언제까지
    # 집계되었는지와는 무관함. 이 컬럼이 후보로 잡히면서 (아직 미래인) 공급승인 예정 단지의
    # 날짜가 "가장 최근 데이터일"로 잘못 선택되어 종료일 기본값이 틀어지는 문제가 있었음.
    exclude_cols = ["공급승인일자"]
    priority_dates = []
    generic_dates = []

    for col in _df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(_df[col]):
            continue
        try:
            s = pd.to_datetime(_df[col], errors="coerce").dropna()
            s = s[(s >= pd.Timestamp('2000-01-01')) & (s <= pd.Timestamp('2100-01-01'))]
            if s.empty:
                continue
            col_l = str(col).lower()
            if any(k in col_l for k in priority_keywords):
                priority_dates.append(s.max())
            else:
                generic_dates.append(s.max())
        except:
            pass

    if priority_dates:
        return max(priority_dates)
    elif generic_dates:
        return max(generic_dates)
    return None

_default_start = pd.Timestamp("2021-01-01").date()

if df is not None and not df.empty:
    _last_ts = _last_data_date_from_df(df)
else:
    _last_ts = None

if _last_ts is not None:
    _default_end = (_last_ts + pd.offsets.MonthEnd(0)).date()
else:
    # 🛠 [버그 수정] 엑셀에 데이터 기준일 컬럼이 없는 경우, 월간 보고는 보통 한 달
    # 지연되어 집계되므로 "이번 달 말"이 아닌 "전월 말"을 기본값으로 사용.
    # (※ 개월 컬럼의 마지막 값 기반 추정은 이 파일 구조와 맞지 않아 되돌림)
    _default_end = (pd.Timestamp.today().replace(day=1) - pd.Timedelta(days=1)).date()

top_container.markdown("#### 분석 기간(연·월 기준)")

start_raw = top_container.date_input("시작일 (연·월 기준)", value=_default_start)
end_raw = top_container.date_input("종료일 (연·월 기준)", value=_default_end)

start_raw = pd.to_datetime(start_raw)
end_raw = pd.to_datetime(end_raw)

시작일 = pd.Timestamp(year=start_raw.year, month=start_raw.month, day=1)
종료일 = pd.Timestamp(year=end_raw.year, month=end_raw.month, day=1) + pd.offsets.MonthEnd(0)

if 시작일 > 종료일:
    시작일, 종료일 = 종료일, 시작일

min_units = top_container.number_input("세대수 하한(세대)", min_value=0, max_value=2000, step=50, value=300)

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if top_container.button("입주율 분석 실행", key="run_btn"):
    st.session_state.run_clicked = True

run = st.session_state.run_clicked

# -------------------- 분석/시각화 --------------------
# -------------------- [이동] 최상단: 연도별 누적 입주율 (기존 분석 함수에서 옮김) --------------------
def show_yearly_cumulative(시작일, 종료일, min_units=0):
    시작일 = pd.to_datetime(시작일)
    종료일 = pd.to_datetime(종료일)
    month_cols = ensure_start_index(df)

    mask = (
        (df["입주시작월"] >= 시작일)
        & (df["입주시작월"] <= 종료일)
        & (df["세대수"].fillna(0) >= min_units)
    )
    base = df.loc[mask & df["입주시작index"].notna()].copy()

    def cum_until_end(row):
        idx = int(row["입주시작index"])
        months_elapsed = (종료일.year - row["공급승인일자"].year) * 12 + (종료일.month - row["공급승인일자"].month)
        end_idx = min(len(month_cols) - 1, months_elapsed)
        cols = month_cols[idx:end_idx + 1]
        vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
        return sum(vals)

    if not base.empty:
        base["입주세대수"] = base.apply(cum_until_end, axis=1)
        base["입주기간(개월)"] = base.apply(
            lambda r: max(0, min(
                len(month_cols) - 1,
                (종료일.year - r["공급승인일자"].year) * 12 + (종료일.month - r["공급승인일자"].month),
            ) - int(r["입주시작index"]) + 1) if pd.notna(r["입주시작index"]) else np.nan,
            axis=1
        )
        base["입주율"] = base.apply(lambda r: _safe_ratio(r["입주세대수"], r["세대수"]), axis=1)
        base["잔여세대수"] = (base["세대수"] - base["입주세대수"]).clip(lower=0)
    else:
        base["입주세대수"] = []
        base["입주기간(개월)"] = []
        base["입주율"] = []
        base["잔여세대수"] = []

    ybase = base.copy()
    if not ybase.empty:
        ybase["입주시작연도"] = pd.to_datetime(ybase["입주시작월"]).dt.year
        yearly = (
            ybase.groupby("입주시작연도")
            .agg(단지수=("아파트명", "count"),
                 총세대수=("세대수", "sum"),
                 총입주세대수=("입주세대수", "sum"))
            .reset_index()
            .sort_values("입주시작연도")
        )
        yearly["잔여세대수"] = (yearly["총세대수"] - yearly["총입주세대수"]).clip(lower=0)
        yearly["누적입주율"] = yearly.apply(lambda r: _safe_ratio(r["총입주세대수"], r["총세대수"]), axis=1)
    else:
        yearly = pd.DataFrame(columns=["입주시작연도","단지수","총세대수","총입주세대수","잔여세대수","누적입주율"])

    st.markdown("#### 📌 연도별 누적 입주율(가중: 총입주세대수 ÷ 총세대수)")
    if not yearly.empty:
        cols = st.columns(len(yearly))
        for col, (_, row) in zip(cols, yearly.iterrows()):
            col.metric(
                label=f"{int(row['입주시작연도'])}년",
                value=f"{row['누적입주율']*100:.1f}%",
                delta=f"{int(row['총입주세대수']):,} / {int(row['총세대수']):,}"
            )
    else:
        st.info("조건에 맞는 연도별 데이터가 없어.")

    yearly_disp = yearly.copy()
    for c in ["단지수","총세대수","총입주세대수","잔여세대수","입주시작연도"]:
        if c in yearly_disp.columns:
            yearly_disp[c] = pd.to_numeric(yearly_disp[c], errors="coerce").round().astype("Int64")
    yearly_disp = _format_pct_cols(yearly_disp, ["누적입주율"])
    st.dataframe(
        yearly_disp,
        use_container_width=True,
        column_config={
            "입주시작연도": st.column_config.NumberColumn("입주시작연도", format="%d"),
            "단지수": st.column_config.NumberColumn("단지수", format="%,d"),
            "총세대수": st.column_config.NumberColumn("총세대수", format="%,d"),
            "총입주세대수": st.column_config.NumberColumn("총입주세대수", format="%,d"),
            "잔여세대수": st.column_config.NumberColumn("잔여세대수", format="%,d"),
            "누적입주율": st.column_config.TextColumn("누적입주율"),
        },
    )


def analyze_occupancy_by_period(시작일, 종료일, min_units=0):
    시작일 = pd.to_datetime(시작일)
    종료일 = pd.to_datetime(종료일)
    month_cols = ensure_start_index(df)

    mask = (
        (df["입주시작월"] >= 시작일)
        & (df["입주시작월"] <= 종료일)
        & (df["세대수"].fillna(0) >= min_units)
    )
    base = df.loc[mask & df["입주시작index"].notna()].copy()

    def cum_until_end(row):
        idx = int(row["입주시작index"])
        months_elapsed = (종료일.year - row["공급승인일자"].year) * 12 + (종료일.month - row["공급승인일자"].month)
        end_idx = min(len(month_cols) - 1, months_elapsed)
        cols = month_cols[idx:end_idx + 1]
        vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
        return sum(vals)

    if not base.empty:
        base["입주세대수"] = base.apply(cum_until_end, axis=1)
        base["입주기간(개월)"] = base.apply(
            lambda r: max(0, min(
                len(month_cols) - 1,
                (종료일.year - r["공급승인일자"].year) * 12 + (종료일.month - r["공급승인일자"].month),
            ) - int(r["입주시작index"]) + 1) if pd.notna(r["입주시작index"]) else np.nan,
            axis=1
        )
        base["입주율"] = base.apply(lambda r: _safe_ratio(r["입주세대수"], r["세대수"]), axis=1)
        base["잔여세대수"] = (base["세대수"] - base["입주세대수"]).clip(lower=0)
    else:
        base["입주세대수"] = []
        base["입주기간(개월)"] = []
        base["입주율"] = []
        base["잔여세대수"] = []

    ybase = base.copy()
    if not ybase.empty:
        ybase["입주시작연도"] = pd.to_datetime(ybase["입주시작월"]).dt.year
        yearly = (
            ybase.groupby("입주시작연도")
            .agg(단지수=("아파트명", "count"),
                 총세대수=("세대수", "sum"),
                 총입주세대수=("입주세대수", "sum"))
            .reset_index()
            .sort_values("입주시작연도")
        )
        yearly["잔여세대수"] = (yearly["총세대수"] - yearly["총입주세대수"]).clip(lower=0)
        yearly["누적입주율"] = yearly.apply(lambda r: _safe_ratio(r["총입주세대수"], r["총세대수"]), axis=1)
    else:
        yearly = pd.DataFrame(columns=["입주시작연도","단지수","총세대수","총입주세대수","잔여세대수","누적입주율"])

    st.markdown("---")

    result_df = (
        base[["아파트명","공급승인일자","세대수","입주시작월",
              "입주세대수","잔여세대수","입주기간(개월)","입주율"]]
        .dropna(subset=["입주세대수"])
        .sort_values(by="공급승인일자", ascending=False)
        .copy()
    )

    display_df = result_df.copy()
    display_df["공급승인일자"] = _fmt_date_str(display_df["공급승인일자"])
    display_df["입주시작월"]   = _fmt_date_str(display_df["입주시작월"])
    for c in ["세대수","입주세대수","잔여세대수","입주기간(개월)"]:
        if c in display_df.columns:
            display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round().astype("Int64")

    # ✅ [수정] 입주율 0.0~1.0 → 0.0~100.0 으로 변환 후 NumberColumn으로 표시 (숫자 정렬 유지)
    display_df["입주율"] = display_df["입주율"].apply(
        lambda x: round(x * 100, 1) if pd.notna(x) else x
    )

    st.subheader(f"✅ [{시작일:%Y-%m} ~ {종료일:%Y-%m}] (세대수 ≥ {min_units}) 입주현황 요약표")
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "공급승인일자": st.column_config.TextColumn("공급승인일자"),
            "입주시작월":   st.column_config.TextColumn("입주시작월"),
            "세대수":       st.column_config.NumberColumn("세대수", format="%,d"),
            "입주세대수":   st.column_config.NumberColumn("입주세대수", format="%,d"),
            "잔여세대수":   st.column_config.NumberColumn("잔여세대수", format="%,d"),
            "입주기간(개월)": st.column_config.NumberColumn("입주기간(개월)", format="%d"),
            # ✅ [수정] 100 기준 숫자로 변환했으므로 %.1f%% 로 올바르게 표시됨
            "입주율":       st.column_config.NumberColumn("입주율", format="%.1f%%"),
        },
    )

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 요약표 CSV 다운로드", data=csv, file_name="occupancy_summary.csv", mime="text/csv")

    return result_df

def plot_yearly_avg_occupancy_with_plan(start_date, end_date, min_units=0):
    month_cols = ensure_start_index(df)
    MAX_M = 9
    start_date = pd.to_datetime(start_date); end_date = pd.to_datetime(end_date)
    cohort = df[
        (df["입주시작월"] >= start_date)
        & (df["입주시작월"] <= end_date)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()
    cohort["입주시작연도"] = cohort["입주시작월"].dt.year

    rate_dict = {}
    has_data = False

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}],
               [{"type": "table"}]],
        row_heights=[0.75, 0.25]
    )

    for y, g in cohort.groupby("입주시작연도"):
        rates = []
        for m in range(1, MAX_M + 1):
            eligible = g[(g["입주시작월"] + pd.offsets.DateOffset(months=m - 1)) <= end_date].copy()
            if eligible.empty:
                rates.append(np.nan); continue
            def cum_n(row, m=m):
                idx = int(row["입주시작index"])
                cols = month_cols[idx: idx + m]
                vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
                return sum(vals)
            num = eligible.apply(cum_n, axis=1).sum()
            den = eligible["세대수"].sum()
            rates.append(_safe_ratio(num, den))
        if any(pd.notna(r) and r > 0 for r in rates): has_data = True
        rate_dict[y] = rates

        fig.add_trace(
            go.Scatter(x=[f"{i}개월" for i in range(1, MAX_M + 1)], y=rates, mode='lines+markers', name=f"{y}년"),
            row=1, col=1
        )

    PLAN = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
    plan_x = list(range(1, MAX_M + 1))
    plan_x_str = [f"{i}개월" for i in plan_x]
    plan_y = [min(1.0, PLAN[i] / 100) for i in plan_x]

    fig.add_trace(
        go.Scatter(x=plan_x_str, y=plan_y, mode='lines+markers', name="사업계획 기준", line=dict(dash='dash', color='magenta')),
        row=1, col=1
    )

    idx_names = plan_x_str
    graph_raw_df = pd.DataFrame(rate_dict, index=idx_names)

    if has_data:
        st.subheader("📈 연도별 입주시작 단지의 월별 누적 입주율")

        table_df = graph_raw_df.T.copy()
        plan_row = pd.DataFrame([plan_y], index=["사업계획 기준"], columns=table_df.columns)
        table_df = pd.concat([table_df, plan_row], axis=0)

        def _fmt_pct(x): return "" if pd.isna(x) else f"{x*100:.1f}%"
        display_df = table_df.map(_fmt_pct)

        header_values = [""] + list(display_df.columns)
        cell_values = [display_df.index.tolist()] + [display_df[c].tolist() for c in display_df.columns]

        fig.add_trace(
            go.Table(
                header=dict(values=header_values, fill_color='lightgray', align='center', font=dict(size=13, color='black')),
                cells=dict(values=cell_values, fill_color='white', align='center', font=dict(size=12, color='black'), height=28)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=None,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=20, b=10),
            height=700
        )

        fig.update_yaxes(title_text="누적 평균 입주율", range=[0, 1], tickformat=".1%", row=1, col=1)
        fig.update_xaxes(title_text="입주경과 개월 (해당 n개월 이상 경과 단지만 포함)", row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ 표시할 연도별 입주율 데이터가 없어.")

def recent2y_top_at_5m(end_date, top_n=10, min_units=0):
    end_date = pd.to_datetime(end_date); month_cols = ensure_start_index(df)
    start_cal = pd.Timestamp(year=end_date.year - 1, month=1, day=1)
    cohort = df[
        (df["입주시작월"] >= start_cal)
        & (df["입주시작월"] <= end_date)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()
    eligible = cohort[(cohort["입주시작월"] + pd.offsets.DateOffset(months=4)) <= end_date].copy()
    if eligible.empty:
        st.info("⚠️ 최근 2년 코호트에서 5개월차까지 도달한 단지가 없어."); return pd.DataFrame()

    def cum_rate(row, m):
        idx = int(row["입주시작index"]); cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    for m in [3, 4, 5]:
        eligible[f"입주율_{m}개월"] = eligible.apply(lambda r, m=m: cum_rate(r, m), axis=1)

    out_cols = ["아파트명","세대수","입주시작월","입주율_3개월","입주율_4개월","입주율_5개월"]
    ranked = eligible[out_cols].sort_values(by="입주율_5개월", ascending=False).reset_index(drop=True)

    disp = ranked.head(top_n).copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])
    disp = _format_pct_cols(disp, ["입주율_3개월","입주율_4개월","입주율_5개월"])

    st.subheader(f"🏆 최근 2년 — 5개월차 입주율 TOP {top_n} (세대수 ≥ {min_units})")
    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "세대수": st.column_config.NumberColumn("세대수", format="%,d"),
            "입주율_3개월": st.column_config.TextColumn("입주율_3개월"),
            "입주율_4개월": st.column_config.TextColumn("입주율_4개월"),
            "입주율_5개월": st.column_config.TextColumn("입주율_5개월"),
        }
    )

    if not ranked.head(top_n).empty:
        if st.toggle("📊 최근 2년 — 5개월차 입주율 차트 보기", value=False):
            fig, ax = plt.subplots(figsize=(4.7, 2.8))
            labels = [f"{n} ({h}세대)" for n, h in zip(ranked.head(top_n)["아파트명"], ranked.head(top_n)["세대수"])]
            ax.barh(labels, ranked.head(top_n)["입주율_5개월"])
            ax.set_xlabel("입주시작 5개월차 입주율", fontsize=8)
            ax.set_title(f"최근 2년 — 5개월차 입주율 TOP (세대수 ≥ {min_units})", fontsize=9)
            ax.tick_params(axis='both', labelsize=7)
            ax.invert_yaxis(); ax.set_xlim(0, 1)
            for y, v in enumerate(ranked.head(top_n)["입주율_5개월"]):
                ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va="center", fontsize=7)
            fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)

    return ranked

def cohort2025_progress(end_date, min_units=0, MAX_M=9):
    end_date = pd.to_datetime(end_date); month_cols = ensure_start_index(df)
    cohort = df[
        (df["입주시작월"] >= pd.Timestamp("2025-01-01"))
        & (df["입주시작월"] <= end_date)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()

    if cohort.empty:
        st.info("⚠️ 2025년 이후 입주시작 단지(조건 충족)가 없어."); return pd.DataFrame()

    def cum_rate(row, m):
        idx = int(row["입주시작index"]); cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    # 🛠 [버그 수정] 실제 경과개월을 MAX_M(=9)로 잘라버리면, MAX_M개월을 넘겨 입주가
    # 진행 중인 단지의 "현재 실적"이 초반 MAX_M개월치 실적으로 축소 왜곡되어 표시됨.
    # (예: 31개월 누적 216세대인 단지가 9개월차 4세대로 표시되는 문제)
    # → 경과개월은 실제 값을 그대로 사용하고, MAX_M은 1~MAX_M 구간 상세표 생성에만 사용.
    def months_elapsed_from_start(row):
        if pd.isna(row["입주시작월"]): return 0
        delta = (end_date.year - row["입주시작월"].year) * 12 + (end_date.month - row["입주시작월"].month) + 1
        return max(0, delta)

    cohort["경과개월(선택일기준)"] = cohort.apply(months_elapsed_from_start, axis=1)
    for m in range(1, MAX_M + 1):
        cohort[f"입주율_{m}개월"] = cohort.apply(lambda r, m=m: cum_rate(r, m) if r["경과개월(선택일기준)"] >= m else np.nan, axis=1)

    # 🛠 [버그 수정] 이전에는 1~MAX_M개월치로만 미리 만들어둔 컬럼(입주율_1개월~입주율_9개월)에서
    # 값을 찾아왔기 때문에 실제 경과개월이 MAX_M을 넘는 단지는 값을 못 찾고 잘못된 값이 됨.
    # → 실제 경과개월(m)로 직접 누적율을 계산하도록 변경.
    def cumulative_as_of_selected(row):
        m = int(row["경과개월(선택일기준)"])
        return np.nan if m <= 0 else cum_rate(row, m)

    cohort["선택일기준_누적입주율"] = cohort.apply(cumulative_as_of_selected, axis=1)

    month_cols_out = [f"입주율_{m}개월" for m in range(1, MAX_M + 1)]
    out_cols = ["아파트명","세대수","입주시작월","경과개월(선택일기준)"] + month_cols_out + ["선택일기준_누적입주율"]
    out_df = cohort[out_cols].sort_values(by="선택일기준_누적입주율", ascending=False)

    disp = out_df.copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])
    disp = _format_pct_cols(disp, month_cols_out + ["선택일기준_누적입주율"])

    st.subheader(f"📊 2025년 이후 입주시작 단지 — 선택일({end_date:%Y-%m-%d}) 기준 누적 입주율 (세대수 ≥ {min_units})")
    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "세대수": st.column_config.NumberColumn("세대수", format="%,d"),
            "경과개월(선택일기준)": st.column_config.NumberColumn("경과개월(선택일기준)", format="%d"),
            **{c: st.column_config.TextColumn(c) for c in month_cols_out + ["선택일기준_누적입주율"]}
        },
    )

    if out_df["선택일기준_누적입주율"].notna().any():
        if st.toggle("📊 2025년 이후 입주시작 단지 누적 입주율 차트 보기", value=False):
            fig, ax = plt.subplots(figsize=(6.7, 4))
            labels = [f"{n} ({h}세대)" for n, h in zip(out_df["아파트명"], out_df["세대수"])]
            ax.barh(labels, out_df["선택일기준_누적입주율"])
            ax.set_xlabel("선택일 기준 누적 입주율", fontsize=9)
            ax.set_title("2025년 이후 입주시작 단지 — 선택일 기준 누적 입주율", fontsize=11)
            ax.tick_params(axis='both', labelsize=8)
            ax.invert_yaxis(); ax.set_xlim(0, 1)
            for y, v in enumerate(out_df["선택일기준_누적입주율"]):
                if pd.notna(v): ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va="center", fontsize=8)
            fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)
    else:
        st.info("⚠️ 선택일 기준 누적입주율을 계산할 수 있는 단지가 없어.")
    return out_df

def underperformers_vs_plan(end_date, min_units=0, MAX_M=9, top_n=15):
    end_date = pd.to_datetime(end_date); month_cols = ensure_start_index(df)

    cohort = df[
        (df["입주시작월"] >= pd.Timestamp("2025-01-01"))
        & (df["입주시작월"] <= end_date)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()

    if cohort.empty:
        st.info("✅ 대상 단지가 없어."); return pd.DataFrame()

    PLAN = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
    PLAN = {k: min(1.0, v / 100) for k, v in PLAN.items()}

    # 🛠 12개월 초과 시 계획을 100%로 간주 (PLAN 원본은 1~9개월치만 존재)
    def _plan_ratio(m):
        if m in PLAN:
            return PLAN[m]
        if m > 12:
            return 1.0
        return np.nan

    def cum_rate(row, m):
        idx = int(row["입주시작index"]); cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    # 🛠 [버그 수정] cohort2025_progress와 동일한 이유로, 실제 경과개월을 MAX_M으로
    # 자르지 않도록 수정. (계획(PLAN) 대비 비교는 원래 1~9개월치만 있으므로,
    # 9개월을 넘는 경우 계획/편차는 자연스럽게 공란(NaN) 처리됨 — 실제 누적실적은 정확히 계산)
    def months_elapsed_from_start(row):
        if pd.isna(row["입주시작월"]): return 0
        delta = (end_date.year - row["입주시작월"].year) * 12 + (end_date.month - row["입주시작월"].month) + 1
        return max(0, delta)

    cohort["경과개월(선택일기준)"] = cohort.apply(months_elapsed_from_start, axis=1)

    actual_list, plan_list, diff_list = [], [], []
    for _, r in cohort.iterrows():
        m = int(r["경과개월(선택일기준)"])
        if m <= 0:
            actual, plan, diff = np.nan, np.nan, np.nan
        else:
            actual = cum_rate(r, m); plan = _plan_ratio(m)
            diff = (actual - plan) if pd.notna(actual) and pd.notna(plan) else np.nan
        actual_list.append(actual); plan_list.append(plan); diff_list.append(diff)

    cohort["실제누적(선택일)"] = actual_list
    cohort["계획누적(선택일)"] = plan_list
    cohort["편차(pp)"] = [(d * 100 if pd.notna(d) else np.nan) for d in diff_list]

    cohort["실제누적세대(선택일)"] = (cohort["실제누적(선택일)"] * cohort["세대수"]).round().astype("Int64")
    cohort["계획누적세대(선택일)"] = (cohort["계획누적(선택일)"] * cohort["세대수"]).round().astype("Int64")
    cohort["현재_부족세대"] = (cohort["계획누적세대(선택일)"] - cohort["실제누적세대(선택일)"]).clip(lower=0).astype("Int64")

    st.markdown("---")
    view_mode = st.radio(
        "📊 조회 대상 선택",
        ["🚨 계획 대비 저조 단지", "🌟 계획 초과(우수 단지)", "전체 단지 보기"],
        horizontal=True,
        key="view_mode_radio"
    )

    if view_mode == "🚨 계획 대비 저조 단지":
        out = cohort[cohort["편차(pp)"] < 0].copy()
        sort_asc = True
        title_prefix = "🚨 2025년 이후 계획 대비 저조 단지"
    elif view_mode == "🌟 계획 초과(우수 단지)":
        out = cohort[cohort["편차(pp)"] >= 0].copy()
        sort_asc = False
        title_prefix = "🌟 2025년 이후 계획 초과(우수) 단지"
    else:
        out = cohort.copy()
        sort_asc = False
        title_prefix = "📊 2025년 이후 전체 단지 계획 대비 실적"

    if out.empty:
        st.info(f"✅ 조건에 맞는 단지가 없어. ({view_mode})"); return pd.DataFrame()

    out = out[
        ["아파트명","세대수","입주시작월","경과개월(선택일기준)",
         "실제누적세대(선택일)","계획누적세대(선택일)","현재_부족세대",
         "실제누적(선택일)","계획누적(선택일)","편차(pp)"]
    ].sort_values(by="편차(pp)", ascending=sort_asc)

    disp_limit = len(out) if view_mode == "전체 단지 보기" else top_n

    disp = out.head(disp_limit).copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])
    disp = _format_pct_cols(disp, ["실제누적(선택일)", "계획누적(선택일)"])

    st.subheader(f"{title_prefix} (선택일 {end_date:%Y-%m-%d}, 세대수 ≥ {min_units}) — 상위 {len(disp)}개")
    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "세대수": st.column_config.NumberColumn("세대수", format="%,d"),
            "경과개월(선택일기준)": st.column_config.NumberColumn("경과개월(선택일기준)", format="%d"),
            "실제누적세대(선택일)": st.column_config.NumberColumn("실제누적세대(선택일)", format="%,d"),
            "계획누적세대(선택일)": st.column_config.NumberColumn("계획누적세대(선택일)", format="%,d"),
            "현재_부족세대": st.column_config.NumberColumn("현재_부족세대", format="%,d"),
            "실제누적(선택일)": st.column_config.TextColumn("실제누적(선택일)"),
            "계획누적(선택일)": st.column_config.TextColumn("계획누적(선택일)"),
            "편차(pp)": st.column_config.NumberColumn("편차(pp)", format="%+.1f"),
        },
    )

    fig_height = max(3.3, len(disp) * 0.35)
    fig, ax = plt.subplots(figsize=(8.7, fig_height))

    worst = out.head(disp_limit).copy()

    y_labels = [f"{n} ({h}세대) · {m}개월차" for n, h, m in zip(worst["아파트명"], worst["세대수"], worst["경과개월(선택일기준)"])]

    ax.barh(y_labels, worst["계획누적세대(선택일)"], height=0.7, color="#AEC6E0", alpha=1.0, edgecolor="none", label="계획 누적 세대")
    ax.barh(y_labels, worst["실제누적세대(선택일)"], height=0.35, color="#1B3358", alpha=1.0, label="실제 누적 세대")

    x_max = max(worst["계획누적세대(선택일)"].max(skipna=True), worst["실제누적세대(선택일)"].max(skipna=True))

    ax.set_xlim(0, float(x_max) * 1.35)
    ax.set_xlabel("누적 세대수", fontsize=9)
    ax.set_title(f"{title_prefix} — 계획 vs 실적 누적 세대수", fontsize=11)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    pad_out = max(8, x_max * 0.015)
    for y, (a, p, lack) in enumerate(zip(
        worst["실제누적세대(선택일)"].fillna(0),
        worst["계획누적세대(선택일)"].fillna(0),
        worst["현재_부족세대"].fillna(0),
    )):
        a = int(a); p = int(p); lack = int(lack)

        if p > a:
            diff_str = f"부족 {lack:,}"
        elif a > p:
            over = a - p
            diff_str = f"초과 {over:,}"
        else:
            diff_str = "계획 달성"

        pos_x = max(a, p) + pad_out
        text_str = f"{a:,}세대 (계획 {p:,} | {diff_str})"
        ax.text(pos_x, y, text_str, va="center", ha="left", fontsize=9, alpha=0.9)

    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4.7))
    scatter_df = worst.dropna(subset=["계획누적(선택일)", "실제누적(선택일)", "편차(pp)"]).copy()
    if scatter_df.empty:
        st.info("⚠️ 산포도에 표시할 값이 없어."); return out
    bubble_area = _bubble_area_from_units(scatter_df["세대수"], min_area=250, max_area=2800)
    sc = ax2.scatter(scatter_df["계획누적(선택일)"], scatter_df["실제누적(선택일)"],
                     s=bubble_area, c=scatter_df["편차(pp)"], alpha=0.9, edgecolors="k", linewidths=0.6)
    ax2.plot([0, 1], [0, 1], "--", linewidth=1)
    xmax = max(1.0, scatter_df["계획누적(선택일)"].max() * 1.05)
    ymax = max(1.0, scatter_df["실제누적(선택일)"].max() * 1.05)
    ax2.set_xlim(0, min(1.0, xmax)); ax2.set_ylim(0, min(1.0, ymax))

    ax2.set_xlabel("계획 누적(비율)", fontsize=9)
    ax2.set_ylabel("실제 누적(비율)", fontsize=9)
    ax2.set_title("계획 vs 실제 (버블=세대수, 색=편차)", fontsize=11)
    ax2.tick_params(axis='both', labelsize=8)

    cb = plt.colorbar(sc)
    cb.set_label("편차(pp)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    for _, r in scatter_df.iterrows():
        ax2.text(float(r["계획누적(선택일)"]) + 0.012, float(r["실제누적(선택일)"]) + 0.012, f"{str(r['아파트명'])}", fontsize=7, alpha=0.95)
    ax2.grid(alpha=0.3); fig2.tight_layout(); apply_korean_font(fig2); st.pyplot(fig2, use_container_width=True)
    return out

# -------------------- [추가] 공동주택 검색 함수 --------------------
def search_complex(keyword: str, ref_date: pd.Timestamp, MAX_M: int = 9):
    """keyword 로 아파트명을 부분검색하여 계획 vs 실적 요약을 표시"""
    month_cols = ensure_start_index(df)
    ref_date = pd.to_datetime(ref_date)

    PLAN = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
    PLAN = {k: min(1.0, v / 100) for k, v in PLAN.items()}

    # 🛠 12개월 초과 시 계획을 100%로 간주 (PLAN 원본은 1~9개월치만 존재)
    def _plan_ratio(m):
        if m in PLAN:
            return PLAN[m]
        if m > 12:
            return 1.0
        return np.nan

    matched = df[df["아파트명"].astype(str).str.contains(keyword, na=False)].copy()

    if matched.empty:
        st.warning(f"🔍 '{keyword}' 와 일치하는 단지가 없습니다.")
        return

    def cum_rate(row, m):
        idx = int(row["입주시작index"])
        cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols])
        den = row["세대수"]
        return _safe_ratio(num, den)

    # 🛠 [버그 수정] 실제 경과개월을 MAX_M(=9)로 잘라버리는 문제가 있었음.
    # 이 때문에 입주시작 후 MAX_M개월을 넘겨 진행 중인 단지(예: 31개월 누적 216세대)의
    # "현재 실적"이 초반 MAX_M개월치 실적(예: 4세대)으로 축소 왜곡되어 표시되었음.
    # → 경과개월은 실제 값을 그대로 사용. PLAN은 원래 1~9개월치만 있으므로 그 이후는
    #   자연스럽게 계획 대비 비교가 생략(NaN)되고, 실제 누적실적은 정확히 계산됨.
    def months_elapsed(row):
        if pd.isna(row.get("입주시작월")):
            return 0
        delta = (ref_date.year - row["입주시작월"].year) * 12 + (ref_date.month - row["입주시작월"].month) + 1
        return max(0, delta)

    rows_out = []
    for _, r in matched.iterrows():
        if pd.isna(r.get("입주시작index")) or pd.isna(r.get("세대수")):
            continue
        m = months_elapsed(r)
        actual = cum_rate(r, m) if m > 0 else np.nan
        plan   = _plan_ratio(m) if m > 0 else np.nan
        diff   = (actual - plan) * 100 if pd.notna(actual) and pd.notna(plan) else np.nan
        actual_units = round(actual * r["세대수"]) if pd.notna(actual) else np.nan
        plan_units   = round(plan   * r["세대수"]) if pd.notna(plan)   else np.nan
        lack_units   = max(0, plan_units - actual_units) if pd.notna(plan_units) and pd.notna(actual_units) else np.nan

        rows_out.append({
            "아파트명":           r["아파트명"],
            "세대수":             int(r["세대수"]),
            "입주시작월":         r.get("입주시작월", pd.NaT),
            "경과개월":           m,
            "실제누적세대":       int(actual_units) if pd.notna(actual_units) else pd.NA,
            "계획누적세대":       int(plan_units)   if pd.notna(plan_units)   else pd.NA,
            "현재_부족세대":      int(lack_units)   if pd.notna(lack_units)   else pd.NA,
            "실제누적(비율)":     actual,
            "계획누적(비율)":     plan,
            "편차(pp)":           diff,
        })

    if not rows_out:
        st.warning(f"🔍 '{keyword}' 단지의 입주시작 데이터가 없습니다.")
        return

    result = pd.DataFrame(rows_out)

    # ── 표 표시 ──
    disp = result.copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])
    disp = _format_pct_cols(disp, ["실제누적(비율)", "계획누적(비율)"])

    st.dataframe(
        disp,
        use_container_width=True,
        column_config={
            "세대수":         st.column_config.NumberColumn("세대수",         format="%,d"),
            "경과개월":       st.column_config.NumberColumn("경과개월",       format="%d"),
            "실제누적세대":   st.column_config.NumberColumn("실제누적세대",   format="%,d"),
            "계획누적세대":   st.column_config.NumberColumn("계획누적세대",   format="%,d"),
            "현재_부족세대":  st.column_config.NumberColumn("현재_부족세대",  format="%,d"),
            "실제누적(비율)": st.column_config.TextColumn("실제누적(비율)"),
            "계획누적(비율)": st.column_config.TextColumn("계획누적(비율)"),
            "편차(pp)":       st.column_config.NumberColumn("편차(pp)",       format="%+.1f"),
        },
    )

    # ── 계획 vs 실적 누적 세대수 가로막대 그래프 ──
    # 🛠 [버그 수정] 계획(PLAN)은 1~9개월치만 존재하므로, 9개월을 넘겨 입주가 진행된
    # 단지는 "계획누적세대"가 없어(None) 이 행이 통째로 제외되어 그래프가 아예
    # 안 보이는 문제가 있었음. → 실제누적세대만 있으면 그리도록 조건 완화.
    plot_df = result.dropna(subset=["실제누적세대"]).copy()
    if plot_df.empty:
        return

    fig_h = max(2.5, len(plot_df) * 0.55) / 4  # 🛠 막대그래프 높이만 추가로 절반 축소(요청 반영)
    fig, ax = plt.subplots(figsize=(8.5, fig_h + 1.1))  # +1.1: 하단 범례 공간(고정, 막대 높이와 무관)

    y_labels = [
        f"{n} ({h}세대) · {m}개월차"
        for n, h, m in zip(plot_df["아파트명"], plot_df["세대수"], plot_df["경과개월"])
    ]

    ax.barh(y_labels, plot_df["계획누적세대"].fillna(0), height=0.7,
            color="#AEC6E0", alpha=1.0, edgecolor="none", label="계획 누적 세대")
    ax.barh(y_labels, plot_df["실제누적세대"].fillna(0), height=0.35,
            color="#1B3358", alpha=1.0, label="실제 누적 세대")

    x_vals = list(plot_df["계획누적세대"]) + list(plot_df["실제누적세대"])
    x_max  = max((v for v in x_vals if pd.notna(v)), default=1)
    ax.set_xlim(0, float(x_max) * 1.40)
    ax.set_title(f"계획 vs 실적 누적 세대수  (기준일: {ref_date:%Y-%m-%d})", fontsize=11)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.42), ncol=2, fontsize=8, frameon=False)

    pad = max(5, x_max * 0.015)
    for yi, (a, p, lack) in enumerate(zip(
        plot_df["실제누적세대"].fillna(0),
        plot_df["계획누적세대"].fillna(0),
        plot_df["현재_부족세대"].fillna(0),
    )):
        a = int(a); p = int(p); lack = int(lack)
        diff_str = f"부족 {lack:,}" if p > a else (f"초과 {a-p:,}" if a > p else "계획 달성")
        ax.text(max(a, p) + pad, yi,
                f"{a:,}세대 (계획 {p:,} | {diff_str})",
                va="center", ha="left", fontsize=8.5, alpha=0.9)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    apply_korean_font(fig)
    st.pyplot(fig, use_container_width=True)

# -------------------- 실행 --------------------
col1, col2 = st.columns([1, 15])

with col1:
    try:
        st.image("logo.png", width=50)
    except Exception:
        pass

with col2:
    st.title("🏡 입주율 분석 대시보드")

st.markdown("##### ✨ Prepared by 마케팅본부 마케팅팀")

# -------------------- 1) 연도별 누적 입주율 (최상단) --------------------
if df is not None and not df.empty:
    show_yearly_cumulative(시작일, 종료일, min_units=min_units)
    st.markdown("---")

# -------------------- [추가] 공동주택 검색 섹션 --------------------
with st.expander("🔍 공동주택 검색", expanded=False):
    if df.empty:
        st.info("데이터를 먼저 불러와 주세요.")
    else:
        s_col1, s_col2 = st.columns([3, 1])
        with s_col1:
            search_keyword = st.text_input(
                "아파트명 검색 (부분 입력 가능)",
                placeholder="예) 해링턴, 힐스테이트, 달서...",
                key="search_keyword_input",
                label_visibility="collapsed",
            )
        with s_col2:
            search_btn = st.button("검색", key="search_btn", use_container_width=True)

        if search_btn and search_keyword.strip():
            ensure_start_index(df)
            ref = 종료일  # 사이드바에서 설정한 종료일 기준
            st.markdown(f"##### 🔎 '{search_keyword.strip()}' 검색 결과 — 기준일: {ref:%Y-%m-%d}")
            search_complex(search_keyword.strip(), ref_date=ref)
        elif search_btn and not search_keyword.strip():
            st.warning("검색어를 입력해 주세요.")
# ── 검색 섹션 끝 ──────────────────────────────────────────────────

# -------------------- 3) 지도 시각화 --------------------
if df is not None and not df.empty:
    st.markdown("#### 🗺️ 공동주택 위치 지도")
    render_main_map(build_top_map_options(시작일, 종료일, min_units), key="top_map")



st.markdown("---")

if chosen_font: st.caption(f"한글 폰트 적용: {chosen_font}")
st.caption(f"{data_caption} | code_ver={CODE_VER}")

if run:
    if df.empty:
        st.error("데이터를 먼저 불러와 주세요.")
    else:
        # 1 & 2. 연도별 누적 입주율 & 입주현황 요약표
        analyze_occupancy_by_period(시작일, 종료일, min_units=min_units)

        # 3. 조회 대상 선택 (계획 대비 저조 단지 등)
        underperformers_vs_plan(종료일, min_units=min_units, MAX_M=9, top_n=15)

        # 4. 연도별 입주시작 단지의 월별 누적 입주율 (그래프, 표)
        plot_yearly_avg_occupancy_with_plan(시작일, 종료일, min_units=min_units)

        # 5. 최근 2년 — 5개월차 입주율 TOP 10 (+활성화 버튼)
        recent2y_top_at_5m(종료일, top_n=10, min_units=min_units)

        # 6. 2025년 이후 입주시작 단지 (+활성화 버튼)
        cohort2025_progress(종료일, min_units=min_units, MAX_M=9)
else:
    st.info("왼쪽 사이드바에서 옵션을 설정하고 **입주율 분석 실행**을 눌러주세요.")
