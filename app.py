# app.py  ─ Streamlit 버전 (한글 폰트 견고화 + 캐시 초기화 버튼 + 파일해시 자동 무효화
#                                 + 최신 파일 자동 선택 + TTL 캐시 + 0~100% 클램프 + CSV 다운로드)
import os, logging, warnings, shutil, time, hashlib, io, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

st.set_page_config(page_title="입주율 분석", layout="wide")

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
        if ax.title: ax.title.set_fontproperties(kprop)
        if ax.xaxis and ax.xaxis.get_label(): ax.xaxis.get_label().set_fontproperties(kprop)
        if ax.yaxis and ax.yaxis.get_label(): ax.yaxis.get_label().set_fontproperties(kprop)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_fontproperties(kprop)
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts(): txt.set_fontproperties(kprop)
        for child in ax.get_children():
            if hasattr(child, "get_celld"):
                for cell in child.get_celld().values():
                    cell._text.set_fontproperties(kprop)

chosen_font = set_korean_font_strict()

# -------------------- 사이드바 --------------------
st.sidebar.markdown("### 데이터 / 필터")

load_way = st.sidebar.radio("데이터 불러오기 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

# 자동 최신 파일 선택
auto_pick_latest = st.sidebar.checkbox("최신 파일 자동 선택(패턴)", value=True)
pattern = st.sidebar.text_input("패턴(자동 선택)", value="data/입주율*.xlsx")

uploaded_file = None
if load_way == "Repo 내 파일 사용":
    excel_path = st.sidebar.text_input("엑셀 파일 경로(수동)", value="data/입주율.xlsx")
else:
    uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=["xlsx"])
    excel_path = None

# 캐시 초기화 버튼
if st.sidebar.button("데이터 캐시 초기화"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("캐시 초기화 완료")
    st.rerun()

# TTL 자동 갱신
ttl_minutes = st.sidebar.number_input("자동 갱신 주기(TTL, 분)", min_value=0, max_value=120, value=0, step=5)

시작일 = st.sidebar.text_input("시작일", value="2021/01/01")
종료일 = st.sidebar.text_input("종료일", value="2025/08/31")
min_units = st.sidebar.number_input("세대수 하한(세대)", min_value=0, max_value=2000, step=50, value=300)
run = st.sidebar.button("입주율 분석 실행", key="run_btn")

# -------------------- 데이터 로드 --------------------
def file_digest_from_path(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()[:10]

def file_digest_from_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()[:10]

def ttl_bucket(minutes: int) -> str:
    if not minutes or minutes <= 0: return "ttl0"
    return f"ttl{int(time.time() // (minutes * 60))}"

@st.cache_data(show_spinner=False)
def load_df_from_path_or_buffer(path_str: str | None, buffer_bytes: bytes | None, digest: str, ttl_key: str):
    if buffer_bytes is not None:
        bio = io.BytesIO(buffer_bytes)
        df_local = pd.read_excel(bio, sheet_name="data")
    else:
        if not path_str or not os.path.exists(path_str):
            return pd.DataFrame()
        df_local = pd.read_excel(path_str, sheet_name="data")
    df_local["공급승인일자"] = pd.to_datetime(df_local["공급승인일자"], errors="coerce")
    return df_local

# Repo 경로 선택(자동/수동)
selected_path_str = None
auto_hint = ""
if load_way == "Repo 내 파일 사용":
    if auto_pick_latest:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths = sorted([Path(p) for p in matches], key=lambda p: p.stat().st_mtime, reverse=True)
            selected_path_str = str(paths[0])
            auto_hint = f"(자동선택: {Path(selected_path_str).name})"
        else:
            selected_path_str = excel_path
            auto_hint = "(패턴 일치 없음 → 수동 경로 사용)"
    else:
        selected_path_str = excel_path

# 파일 경로/업로드에 따라 digest 생성 → 캐시 자동 무효화
data_caption = ""
ttl_key = ttl_bucket(ttl_minutes)

if load_way == "Repo 내 파일 사용":
    p = Path(selected_path_str) if selected_path_str else None
    if p and p.exists():
        digest = file_digest_from_path(p)
        df = load_df_from_path_or_buffer(str(p), None, digest, ttl_key)
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        data_caption = f"📄 Source=Repo:{p.name} {auto_hint} | ver={digest} | updated={mtime} | {ttl_key}"
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

# -------------------- 분석 함수들 --------------------
def _fmt_date_str(series):
    """표시용 YYYY-MM-DD 문자열로 변환(계산용 원본은 유지)"""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

def analyze_occupancy_by_period(시작일, 종료일, min_units=0):
    """첫 번째 요약표: 날짜 YYYY-MM-DD, 잔여세대수 추가, 연도별 누적 입주율 하단 표기"""
    시작일 = pd.to_datetime(시작일); 종료일 = pd.to_datetime(종료일)
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
        cols = month_cols[idx : end_idx + 1]
        vals = [0 if pd.isna(row[c]) else row[c] for c in cols]
        return sum(vals)

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

    result_df = (
        base[["아파트명", "공급승인일자", "세대수", "입주시작월",
              "입주세대수", "잔여세대수", "입주기간(개월)", "입주율"]]
        .dropna(subset=["입주세대수"])
        .sort_values(by="입주율", ascending=False)
        .copy()
    )

    # 표시용: 날짜 문자열화
    display_df = result_df.copy()
    display_df["공급승인일자"] = _fmt_date_str(display_df["공급승인일자"])
    display_df["입주시작월"]   = _fmt_date_str(display_df["입주시작월"])

    st.subheader(f"✅ [{시작일:%Y-%m-%d} ~ {종료일:%Y-%m-%d}] (세대수 ≥ {min_units}) 입주현황 요약표")
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "공급승인일자": st.column_config.TextColumn("공급승인일자"),
            "입주시작월":   st.column_config.TextColumn("입주시작월"),
            "세대수":      st.column_config.NumberColumn("세대수", format="%,d"),
            "입주세대수":   st.column_config.NumberColumn("입주세대수", format="%,d"),
            "잔여세대수":   st.column_config.NumberColumn("잔여세대수", format="%,d"),
            "입주기간(개월)": st.column_config.NumberColumn("입주기간(개월)", format="%d"),
            "입주율":      st.column_config.NumberColumn("입주율", format="%.1f%%"),
        },
    )

    # CSV 다운로드(원본 유지)
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 요약표 CSV 다운로드", data=csv, file_name="occupancy_summary.csv", mime="text/csv")

    # 연도별 누적(가중) 입주율 표
    ybase = base.copy()
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

    st.markdown("#### 📌 연도별 누적 입주율 (가중: 총입주세대수 ÷ 총세대수)")
    st.dataframe(
        yearly.assign(누적입주율=lambda d: d["누적입주율"] * 100),
        use_container_width=True,
        column_config={
            "입주시작연도": st.column_config.NumberColumn("입주시작연도", format="%d"),
            "단지수": st.column_config.NumberColumn("단지수", format="%,d"),
            "총세대수": st.column_config.NumberColumn("총세대수", format="%,d"),
            "총입주세대수": st.column_config.NumberColumn("총입주세대수", format="%,d"),
            "잔여세대수": st.column_config.NumberColumn("잔여세대수", format="%,d"),
            "누적입주율": st.column_config.NumberColumn("누적입주율(%)", format="%.1f%%"),
        },
    )
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
    fig = plt.figure(figsize=(14, 12), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1.6])
    ax_plot = fig.add_subplot(gs[0]); ax_table = fig.add_subplot(gs[1]); ax_table.axis("off")

    has_data = False
    for y, g in cohort.groupby("입주시작연도"):
        rates = []
        for m in range(1, MAX_M + 1):
            eligible = g[(g["입주시작월"] + pd.offsets.DateOffset(months=m - 1)) <= end_date].copy()
            if eligible.empty:
                rates.append(np.nan); continue
            def cum_n(row, m=m):
                idx = int(row["입주시작index"])
                cols = month_cols[idx : idx + m]
                vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
                return sum(vals)
            num = eligible.apply(cum_n, axis=1).sum()
            den = eligible["세대수"].sum()
            rates.append(_safe_ratio(num, den))
        if any(pd.notna(r) and r > 0 for r in rates): has_data = True
        rate_dict[y] = rates
        ax_plot.plot(range(1, MAX_M + 1), rates, marker="o", label=f"{y}년")

    PLAN = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
    plan_x = list(range(1, MAX_M + 1))
    plan_y = [min(1.0, PLAN[i] / 100) for i in plan_x]
    ax_plot.plot(plan_x, plan_y, linestyle="--", marker="x", label="사업계획 기준")

    idx_names = [f"{i}개월" for i in plan_x]
    graph_raw_df = pd.DataFrame(rate_dict, index=idx_names)

    if has_data:
        ax_plot.set_title(f"연도별 입주시작 단지의 월별 누적 입주율(세대수 ≥ {min_units})")
        ax_plot.set_xlabel("입주경과 개월\n(해당 n개월 이상 경과 단지만 포함)")
        ax_plot.set_ylabel("누적 평균 입주율")
        ax_plot.set_xticks(plan_x, [f"{i}개월" for i in plan_x])
        ax_plot.set_ylim(0, 1); ax_plot.grid(True); ax_plot.legend(ncol=3, loc="lower right")

        table_df = graph_raw_df.T.copy()
        plan_row = pd.DataFrame([plan_y], index=["사업계획 기준"], columns=table_df.columns)
        table_df = pd.concat([table_df, plan_row], axis=0)

        def _fmt_pct(x): return "" if pd.isna(x) else f"{x*100:.1f}%"
        display_df = table_df.applymap(_fmt_pct)
        t = ax_table.table(cellText=display_df.values,
                           rowLabels=display_df.index.tolist(),
                           colLabels=display_df.columns.tolist(),
                           cellLoc="center", loc="center")
        t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 1.7)
        plt.subplots_adjust(hspace=0.28)
        apply_korean_font(fig); st.pyplot(fig, use_container_width=True)
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
        idx = int(row["입주시작index"]); cols = month_cols[idx : idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    for m in [3, 4, 5]:
        eligible[f"입주율_{m}개월"] = eligible.apply(lambda r, m=m: cum_rate(r, m), axis=1)

    out_cols = ["아파트명", "세대수", "입주시작월", "입주율_3개월", "입주율_4개월", "입주율_5개월"]
    ranked = eligible[out_cols].sort_values(by="입주율_5개월", ascending=False).reset_index(drop=True)

    # 표시용: 날짜 문자열화
    disp = ranked.head(top_n).copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])

    st.subheader(f"🏆 최근 2년 — 5개월차 입주율 TOP {top_n} (세대수 ≥ {min_units})")
    st.dataframe(
        disp.style.format({"입주율_3개월": "{:.1%}", "입주율_4개월": "{:.1%}", "입주율_5개월": "{:.1%}"}),
        use_container_width=True
    )

    if not ranked.head(top_n).empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{n} ({h}세대)" for n, h in zip(ranked.head(top_n)["아파트명"], ranked.head(top_n)["세대수"])]
        ax.barh(labels, ranked.head(top_n)["입주율_5개월"])
        ax.set_xlabel("입주시작 5개월차 입주율"); ax.set_title(f"최근 2년 — 5개월차 입주율 TOP (세대수 ≥ {min_units})")
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(ranked.head(top_n)["입주율_5개월"]):
            ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va="center")
        fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)
    return ranked

def cohort2025_progress(end_date, min_units=0, MAX_M=9):
    end_date = pd.to_datetime(end_date); month_cols = ensure_start_index(df)
    cohort = df[
        (df["입주시작월"].dt.year == 2025)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()
    if cohort.empty:
        st.info("⚠️ 2025년 입주시작 단지(조건 충족)가 없어."); return pd.DataFrame()

    def cum_rate(row, m):
        idx = int(row["입주시작index"]); cols = month_cols[idx : idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    def months_elapsed_from_start(row):
        if pd.isna(row["입주시작월"]): return 0
        delta = (end_date.year - row["입주시작월"].year) * 12 + (end_date.month - row["입주시작월"].month) + 1
        return max(0, min(MAX_M, delta))

    cohort["경과개월(선택일기준)"] = cohort.apply(months_elapsed_from_start, axis=1)
    for m in range(1, MAX_M + 1):
        cohort[f"입주율_{m}개월"] = cohort.apply(lambda r, m=m: cum_rate(r, m) if r["경과개월(선택일기준)"] >= m else np.nan, axis=1)

    def cumulative_as_of_selected(row):
        m = int(row["경과개월(선택일기준)"])
        return np.nan if m <= 0 else row.get(f"입주율_{m}개월", np.nan)

    cohort["선택일기준_누적입주율"] = cohort.apply(cumulative_as_of_selected, axis=1)

    month_cols_out = [f"입주율_{m}개월" for m in range(1, MAX_M + 1)]
    out_cols = ["아파트명", "세대수", "입주시작월", "경과개월(선택일기준)"] + month_cols_out + ["선택일기준_누적입주율"]
    out_df = cohort[out_cols].sort_values(by="선택일기준_누적입주율", ascending=False)

    # 표시용: 날짜 문자열화
    disp = out_df.copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])

    st.subheader(f"📊 2025년 입주시작 단지 — 선택일({end_date:%Y-%m-%d}) 기준 누적 입주율 (세대수 ≥ {min_units})")
    st.dataframe(
        disp.style.format({c: "{:.1%}" for c in month_cols_out + ["선택일기준_누적입주율"]}),
        use_container_width=True
    )

    if out_df["선택일기준_누적입주율"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{n} ({h}세대)" for n, h in zip(out_df["아파트명"], out_df["세대수"])]
        ax.barh(labels, out_df["선택일기준_누적입주율"])
        ax.set_xlabel("선택일 기준 누적 입주율"); ax.set_title("2025년 입주시작 단지 — 선택일 기준 누적 입주율")
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(out_df["선택일기준_누적입주율"]):
            if pd.notna(v): ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va="center")
        fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)
    else:
        st.info("⚠️ 선택일 기준 누적입주율을 계산할 수 있는 단지가 없어.")
    return out_df

def underperformers_vs_plan(end_date, min_units=0, MAX_M=9, top_n=15):
    end_date = pd.to_datetime(end_date); month_cols = ensure_start_index(df)
    cohort = df[
        (df["입주시작월"].dt.year == 2025)
        & (df["입주시작index"].notna())
        & (df["세대수"].notna())
        & (df["세대수"] >= min_units)
    ].copy()
    if cohort.empty:
        st.info("✅ 계획 대비 저조한 단지가 없어."); return pd.DataFrame()

    PLAN = {1: 9.29, 2: 43.25, 3: 62.75, 4: 72.61, 5: 78.17, 6: 81.56, 7: 84.28, 8: 86.07, 9: 87.86}
    PLAN = {k: min(1.0, v / 100) for k, v in PLAN.items()}

    def cum_rate(row, m):
        idx = int(row["입주시작index"]); cols = month_cols[idx : idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols]); den = row["세대수"]
        return _safe_ratio(num, den)

    def months_elapsed_from_start(row):
        if pd.isna(row["입주시작월"]): return 0
        delta = (end_date.year - row["입주시작월"].year) * 12 + (end_date.month - row["입주시작월"].month) + 1
        return max(0, min(MAX_M, delta))

    cohort["경과개월(선택일기준)"] = cohort.apply(months_elapsed_from_start, axis=1)

    actual_list, plan_list, diff_list = [], [], []
    for _, r in cohort.iterrows():
        m = int(r["경과개월(선택일기준)"])
        if m <= 0:
            actual, plan, diff = np.nan, np.nan, np.nan
        else:
            actual = cum_rate(r, m); plan = PLAN.get(m, np.nan)
            diff = (actual - plan) if pd.notna(actual) and pd.notna(plan) else np.nan
        actual_list.append(actual); plan_list.append(plan); diff_list.append(diff)

    cohort["실제누적(선택일)"] = actual_list
    cohort["계획누적(선택일)"] = plan_list
    cohort["편차(pp)"] = [(d * 100 if pd.notna(d) else np.nan) for d in diff_list]

    cohort["실제누적세대(선택일)"] = (cohort["실제누적(선택일)"] * cohort["세대수"]).round().astype("Int64")
    cohort["계획누적세대(선택일)"] = (cohort["계획누적(선택일)"] * cohort["세대수"]).round().astype("Int64")
    cohort["현재_부족세대"] = (cohort["계획누적세대(선택일)"] - cohort["실제누적세대(선택일)"]).clip(lower=0).astype("Int64")

    out = cohort[cohort["편차(pp)"] < 0].copy()
    if out.empty:
        st.info("✅ 계획 대비 저조한 단지가 없어."); return pd.DataFrame()

    out = out[
        ["아파트명","세대수","입주시작월","경과개월(선택일기준)",
         "실제누적세대(선택일)","계획누적세대(선택일)","현재_부족세대",
         "실제누적(선택일)","계획누적(선택일)","편차(pp)"]
    ].sort_values(by="편차(pp)", ascending=True)

    # 표시용: 날짜 문자열화
    disp = out.head(top_n).copy()
    disp["입주시작월"] = _fmt_date_str(disp["입주시작월"])

    st.subheader(f"🚨 계획 대비 저조 단지 (선택일 {end_date:%Y-%m-%d}, 세대수 ≥ {min_units}) — 상위 {top_n}개")
    st.dataframe(
        disp.style.format(
            {"실제누적(선택일)": "{:.1%}", "계획누적(선택일)": "{:.1%}",
             "실제누적세대(선택일)": "{:,.0f}", "계획누적세대(선택일)": "{:,.0f}",
             "현재_부족세대": "{:,.0f}", "편차(pp)": "{:+.1f}"}
        ),
        use_container_width=True
    )

    # 시각화는 원본 out 사용
    fig, ax = plt.subplots(figsize=(13, 5))
    worst = out.head(top_n).copy()
    y_labels = [f"{n} ({h}세대) · {m}개월차" for n, h, m in zip(worst["아파트명"], worst["세대수"], worst["경과개월(선택일기준)"])]
    ax.barh(y_labels, worst["계획누적세대(선택일)"], alpha=0.55, edgecolor="none", label="계획 누적 세대")
    ax.barh(y_labels, worst["실제누적세대(선택일)"], alpha=0.95, label="실제 누적 세대")
    ax.autoscale_view(); x_min, x_max = ax.get_xlim(); span = x_max - x_min

    for y, (a_units, p_units) in enumerate(
        zip(worst["실제누적세대(선택일)"].astype(float),
            worst["계획누적세대(선택일)"].astype(float))):
        if np.isfinite(a_units): ax.text(min(a_units + 3, x_max*0.98), y, f"{int(a_units)}세대", va="center", fontsize=9)

    ax.set_xlabel("누적 세대수"); ax.set_title("계획 대비 저조 단지 — 계획 vs 실적 누적 세대수")
    ax.invert_yaxis(); ax.legend(loc="lower right", ncol=2); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); apply_korean_font(fig); st.pyplot(fig, use_container_width=True)

    # 산포도
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    scatter_df = worst.dropna(subset=["계획누적(선택일)", "실제누적(선택일)", "편차(pp)"]).copy()
    if scatter_df.empty:
        st.info("⚠️ 산포도에 표시할 값이 없어(계획/실제 누적 비율 NaN)."); return out
    bubble_area = _bubble_area_from_units(scatter_df["세대수"], min_area=250, max_area=2800)
    sc = ax2.scatter(scatter_df["계획누적(선택일)"], scatter_df["실제누적(선택일)"],
                     s=bubble_area, c=scatter_df["편차(pp)"], alpha=0.9, edgecolors="k", linewidths=0.6)
    ax2.plot([0, 1], [0, 1], "--", linewidth=1)
    xmax = max(1.0, scatter_df["계획누적(선택일)"].max() * 1.05)
    ymax = max(1.0, scatter_df["실제누적(선택일)"].max() * 1.05)
    ax2.set_xlim(0, min(1.0, xmax)); ax2.set_ylim(0, min(1.0, ymax))
    ax2.set_xlabel("계획 누적(비율)"); ax2.set_ylabel("실제 누적(비율)"); ax2.set_title("계획 vs 실제 (버블=세대수, 색=편차)")
    cb = plt.colorbar(sc); cb.set_label("편차(pp)")
    for _, r in scatter_df.iterrows():
        ax2.text(float(r["계획누적(선택일)"]) + 0.012, float(r["실제누적(선택일)"]) + 0.012, f"{str(r['아파트명'])}", fontsize=10, alpha=0.95)
    ax2.grid(alpha=0.3); fig2.tight_layout(); apply_korean_font(fig2); st.pyplot(fig2, use_container_width=True)
    return out

# -------------------- 실행 --------------------
st.title("입주율 분석 대시보드")
if chosen_font: st.caption(f"한글 폰트 적용: {chosen_font}")
st.caption(data_caption)  # 현재 사용 중인 데이터 버전/수정시각/TTL 버킷

if run:
    if df.empty:
        st.error("데이터를 먼저 불러와 주세요.")
    else:
        analyze_occupancy_by_period(시작일, 종료일, min_units=min_units)   # ← 모든 표 날짜 문자열화 반영
        plot_yearly_avg_occupancy_with_plan(시작일, 종료일, min_units=min_units)
        recent2y_top_at_5m(종료일, top_n=10, min_units=min_units)
        cohort2025_progress(종료일, min_units=min_units, MAX_M=9)
        underperformers_vs_plan(종료일, min_units=min_units, MAX_M=9, top_n=15)
else:
    st.info("왼쪽 사이드바에서 옵션을 설정하고 **입주율 분석 실행**을 눌러줘.")
