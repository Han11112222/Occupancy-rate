# app.py — Streamlit 변환 완성본 (Colab 기능 100% 이식)
import os, io, sys, platform, warnings, logging
from datetime import date
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# =========================
# 0) 페이지/레이아웃
# =========================
st.set_page_config(page_title="입주율 대시보드", layout="wide")

# =========================
# 1) 한글 폰트 설정 + 경고 억제
# =========================
def set_korean_font_strict():
    candidates = [
        os.path.join("fonts", "NanumGothic-Regular.ttf"),     # repo 동봉 시
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",    # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/Library/Fonts/AppleGothic.ttf",                     # mac
        "C:/Windows/Fonts/malgun.ttf",                        # Windows
    ]
    chosen_name = None
    for path in candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
            prop = fm.FontProperties(fname=path)
            chosen_name = prop.get_name()
            break
    if chosen_name:
        mpl.rcParams['font.family'] = chosen_name
        mpl.rcParams['font.sans-serif'] = [chosen_name, 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

set_korean_font_strict()

# =========================
# 2) 데이터 불러오기 (업로드/경로)
# =========================
DEFAULT_PATH = os.path.join("data", "입주율.xlsx")

st.sidebar.header("데이터 / 필터")
data_mode = st.sidebar.radio("데이터 불러오기 방식", ["Repo 내 파일 사용", "파일 업로드"], horizontal=True)

if data_mode == "파일 업로드":
    upload = st.sidebar.file_uploader("입주율.xlsx 업로드 (시트명: data)", type=["xlsx"])
    path_input = None
else:
    upload = None
    path_input = st.sidebar.text_input("엑셀 파일 경로", DEFAULT_PATH)

@st.cache_data(show_spinner=True)
def _read_excel_from_path(path: str) -> pd.DataFrame:
    # 시트명이 'data'가 아니어도 첫 시트 자동 처리
    try:
        return pd.read_excel(path, sheet_name="data", engine="openpyxl")
    except Exception:
        xls = pd.ExcelFile(path, engine="openpyxl")
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

@st.cache_data(show_spinner=True)
def _read_excel_from_bytes(content: bytes) -> pd.DataFrame:
    with io.BytesIO(content) as bio:
        try:
            return pd.read_excel(bio, sheet_name="data", engine="openpyxl")
        except Exception:
            bio.seek(0)
            xls = pd.ExcelFile(bio, engine="openpyxl")
            bio.seek(0)
            return pd.read_excel(io.BytesIO(content), sheet_name=xls.sheet_names[0])

def _postload(df: pd.DataFrame) -> pd.DataFrame:
    if "공급승인일자" in df.columns:
        df["공급승인일자"] = pd.to_datetime(df["공급승인일자"], errors="coerce")
    return df

if data_mode == "파일 업로드":
    if upload is None:
        st.info("좌측에서 **입주율.xlsx**를 업로드하면 시작돼.")
        st.stop()
    df = _postload(_read_excel_from_bytes(upload.getvalue()))
else:
    if not os.path.exists(path_input):
        st.error(f"엑셀 파일을 찾을 수 없어: {path_input}")
        st.stop()
    df = _postload(_read_excel_from_path(path_input))

# =========================
# 3) 공통 유틸
# =========================
def ensure_start_index(_df: pd.DataFrame):
    month_cols = [c for c in _df.columns if '개월' in str(c)]
    def _key(c):
        s = ''.join(ch for ch in str(c) if ch.isdigit())
        return int(s) if s else 0
    month_cols = sorted(month_cols, key=_key)

    def first_valid_idx(row):
        for i, col in enumerate(month_cols):
            v = row.get(col, np.nan)
            if pd.notna(v) and v > 0:
                return i
        return np.nan

    if '입주시작index' not in _df.columns:
        _df['입주시작index'] = _df.apply(first_valid_idx, axis=1)

    _df['입주시작월'] = _df.apply(
        lambda r: r['공급승인일자'] + pd.DateOffset(months=int(r['입주시작index']))
        if pd.notna(r.get('입주시작index')) and pd.notna(r.get('공급승인일자')) else pd.NaT,
        axis=1
    )
    return month_cols

PLAN = {1:9.29, 2:43.25, 3:62.75, 4:72.61, 5:78.17, 6:81.56, 7:84.28, 8:86.07, 9:87.86}
PLAN_RATIO = {k: v/100 for k, v in PLAN.items()}
MAX_M = 9

# =========================
# 4) 분석 함수들 (Streamlit 출력)
# =========================
def analyze_occupancy_by_period(시작일, 종료일, min_units=0):
    시작일 = pd.to_datetime(시작일); 종료일 = pd.to_datetime(종료일)
    month_cols = ensure_start_index(df)

    mask = (
        (df['입주시작월'] >= 시작일) &
        (df['입주시작월'] <= 종료일) &
        (df['세대수'].fillna(0) >= min_units)
    )
    base = df.loc[mask & df['입주시작index'].notna()].copy()

    def cum_until_end(row):
        idx = int(row['입주시작index'])
        months_elapsed = (종료일.year - row['공급승인일자'].year) * 12 + (종료일.month - row['공급승인일자'].month)
        end_idx = min(len(month_cols) - 1, months_elapsed)
        cols = month_cols[idx: end_idx + 1]
        vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
        return sum(vals)

    base['입주세대수'] = base.apply(cum_until_end, axis=1)
    base['입주기간(개월)'] = base.apply(
        lambda r: max(0,
                      min(len(month_cols)-1,
                          (종료일.year - r['공급승인일자'].year)*12 + (종료일.month - r['공급승인일자'].month))
                      - int(r['입주시작index']) + 1)
        if pd.notna(r['입주시작index']) else np.nan,
        axis=1
    )
    base['입주율'] = base['입주세대수'] / base['세대수']

    st.markdown(f"**사용된 데이터 기간:** {시작일:%Y-%m-%d} ~ {종료일:%Y-%m-%d} (세대수 ≥ {min_units})")
    result_df = base[['아파트명','공급승인일자','세대수','입주시작월','입주세대수','입주기간(개월)','입주율']] \
                  .dropna(subset=['입주세대수']) \
                  .sort_values(by='입주율', ascending=False)
    st.dataframe(result_df, use_container_width=True)
    return result_df

def plot_yearly_avg_occupancy_with_plan(start_date, end_date, min_units=0):
    month_cols = ensure_start_index(df)
    MAX_M = 9
    start_date = pd.to_datetime(start_date); end_date = pd.to_datetime(end_date)

    cohort = df[
        (df['입주시작월'] >= start_date) &
        (df['입주시작월'] <= end_date) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()
    cohort['입주시작연도'] = cohort['입주시작월'].dt.year

    rate_dict = {}
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.8])
    ax_plot = fig.add_subplot(gs[0]); ax_table = fig.add_subplot(gs[1]); ax_table.axis('off')

    has_data = False
    for y, g in cohort.groupby('입주시작연도'):
        rates = []
        for m in range(1, MAX_M+1):
            eligible = g[(g['입주시작월'] + pd.offsets.DateOffset(months=m-1)) <= end_date].copy()
            if eligible.empty:
                rates.append(np.nan); continue
            def cum_n(row, m=m):
                idx = int(row['입주시작index'])
                cols = month_cols[idx: idx + m]
                vals = [0 if pd.isna(row.get(c)) else row.get(c) for c in cols]
                return sum(vals)
            num = eligible.apply(cum_n, axis=1).sum(); den = eligible['세대수'].sum()
            rates.append(num / den if den > 0 else np.nan)
        rate_dict[y] = rates
        if any(pd.notna(r) and r > 0 for r in rates):
            has_data = True; ax_plot.plot(range(1, MAX_M+1), rates, marker='o', label=f'{y}년')

    # 계획선
    plan_x = list(range(1, MAX_M+1)); plan_y = [PLAN_RATIO[i] for i in plan_x]
    ax_plot.plot(plan_x, plan_y, linestyle='--', marker='x', label='사업계획 기준')

    idx_names = [f'{i}개월' for i in plan_x]
    graph_raw_df = pd.DataFrame(rate_dict, index=idx_names)

    if has_data:
        ax_plot.set_title(f"연도별 입주시작 단지의 월별 누적 입주율(세대수 ≥ {min_units})\n기간: {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}")
        ax_plot.set_xlabel("입주경과 개월(1~9)")
        ax_plot.set_ylabel("누적 평균 입주율")
        ax_plot.set_xticks(plan_x, [f"{i}개월" for i in plan_x])
        ax_plot.set_ylim(0, 1); ax_plot.grid(True); ax_plot.legend(ncol=3, loc='lower right')

        # 표(계획행 포함, %표시)
        table_df = graph_raw_df.T
        plan_row = pd.DataFrame([plan_y], index=['사업계획 기준'], columns=table_df.columns)
        table_df = pd.concat([table_df, plan_row], axis=0)

        def pct1(x): return "" if pd.isna(x) else f"{x*100:.1f}%"
        cell_text = table_df.applymap(pct1).values
        t = ax_table.table(
            cellText=cell_text,
            rowLabels=table_df.index.tolist(),
            colLabels=table_df.columns.tolist(),
            cellLoc='center', loc='center'
        )
        t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1.1, 1.9)

    st.pyplot(fig, clear_figure=True)

def recent2y_top_at_5m(end_date, top_n=10, min_units=0):
    end_date = pd.to_datetime(end_date)
    month_cols = ensure_start_index(df)

    start_cal = pd.Timestamp(year=end_date.year - 1, month=1, day=1)
    st.markdown(f"**최근 2년 TOP(5개월차)** — 사용 기간: {start_cal:%Y-%m-%d} ~ {end_date:%Y-%m-%d}")

    cohort = df[
        (df['입주시작월'] >= start_cal) &
        (df['입주시작월'] <= end_date) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()

    eligible = cohort[(cohort['입주시작월'] + pd.offsets.DateOffset(months=4)) <= end_date].copy()
    if eligible.empty:
        st.info("최근 2년 코호트에서 5개월차까지 도달한 단지가 없어요.")
        return pd.DataFrame()

    def cum_rate(row, m):
        idx = int(row['입주시작index'])
        cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols])
        den = row['세대수']
        return (num / den) if den and den > 0 else np.nan

    for m in [3, 4, 5]:
        eligible[f'입주율_{m}개월'] = eligible.apply(lambda r, m=m: cum_rate(r, m), axis=1)

    out_cols = ['아파트명', '세대수', '입주시작월', '입주율_3개월', '입주율_4개월', '입주율_5개월']
    ranked = eligible[out_cols].sort_values(by='입주율_5개월', ascending=False).reset_index(drop=True)

    st.dataframe(ranked.head(top_n).style.format({
        '입주율_3개월':'{:.1%}', '입주율_4개월':'{:.1%}', '입주율_5개월':'{:.1%}'
    }), use_container_width=True)

    top = ranked.head(top_n).copy()
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = [f"{n} ({h}세대)" for n, h in zip(top['아파트명'], top['세대수'])]
        ax.barh(labels, top['입주율_5개월'])
        ax.set_xlabel('입주시작 5개월차 입주율'); ax.set_title(f'최근 2년 — 5개월차 입주율 TOP (세대수 ≥ {min_units})')
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(top['입주율_5개월']):
            ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va='center')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    return ranked

def cohort2025_progress(end_date, min_units=0, MAX_M=9):
    end_date = pd.to_datetime(end_date)
    month_cols = ensure_start_index(df)

    cohort = df[
        (df['입주시작월'].dt.year == 2025) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()

    if cohort.empty:
        st.info("2025년 입주시작 단지가 없어요.")
        return pd.DataFrame()

    def cum_rate(row, m):
        idx = int(row['입주시작index'])
        cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols])
        den = row['세대수']
        return (num / den) if den and den > 0 else np.nan

    def months_elapsed_from_start(row):
        if pd.isna(row['입주시작월']):
            return 0
        delta = (end_date.year - row['입주시작월'].year) * 12 + (end_date.month - row['입주시작월'].month) + 1
        return max(0, min(MAX_M, delta))

    cohort['경과개월(선택일기준)'] = cohort.apply(months_elapsed_from_start, axis=1)

    for m in range(1, MAX_M + 1):
        cohort[f'입주율_{m}개월'] = cohort.apply(
            lambda r, m=m: cum_rate(r, m) if r['경과개월(선택일기준)'] >= m else np.nan, axis=1
        )

    def cumulative_as_of_selected(row):
        m = int(row['경과개월(선택일기준)'])
        if m <= 0: return np.nan
        return row.get(f'입주율_{m}개월', np.nan)

    cohort['선택일기준_누적입주율'] = cohort.apply(cumulative_as_of_selected, axis=1)

    month_cols_out = [f'입주율_{m}개월' for m in range(1, MAX_M + 1)]
    out_cols = ['아파트명', '세대수', '입주시작월', '경과개월(선택일기준)'] + month_cols_out + ['선택일기준_누적입주율']
    out_df = cohort[out_cols].sort_values(by='선택일기준_누적입주율', ascending=False)

    st.markdown(f"**2025 코호트 — 사용 기간:** 2025-01-01 ~ {end_date:%Y-%m-%d}")
    st.dataframe(out_df.style.format({c:'{:.1%}' for c in month_cols_out + ['선택일기준_누적입주율']}), use_container_width=True)

    if out_df['선택일기준_누적입주율'].notna().any():
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = [f"{n} ({h}세대)" for n, h in zip(out_df['아파트명'], out_df['세대수'])]
        ax.barh(labels, out_df['선택일기준_누적입주율'])
        ax.set_xlabel('선택일 기준 누적 입주율')
        ax.set_title('2025년 입주시작 단지 — 선택일 기준 누적 입주율')
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(out_df['선택일기준_누적입주율']):
            if pd.notna(v):
                ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va='center')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("선택일 기준 누적입주율을 계산할 수 있는 단지가 없어요.")
    return out_df

def underperformers_vs_plan(end_date, min_units=0, MAX_M=9, top_n=15):
    end_date = pd.to_datetime(end_date)
    month_cols = ensure_start_index(df)

    cohort = df[
        (df['입주시작월'].dt.year == 2025) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()
    if cohort.empty:
        st.info("2025년 입주시작 단지가 없어요.")
        return pd.DataFrame()

    PLAN_loc = {k: v/100 for k, v in PLAN.items()}

    def cum_rate(row, m):
        idx = int(row['입주시작index'])
        cols = month_cols[idx: idx + m]
        num = sum([0 if pd.isna(row.get(c)) else row.get(c) for c in cols])
        den = row['세대수']
        return (num / den) if den and den > 0 else np.nan

    def months_elapsed_from_start(row):
        if pd.isna(row['입주시작월']):
            return 0
        delta = (end_date.year - row['입주시작월'].year) * 12 + (end_date.month - row['입주시작월'].month) + 1
        return max(0, min(MAX_M, delta))

    cohort['경과개월(선택일기준)'] = cohort.apply(months_elapsed_from_start, axis=1)

    actual_list, plan_list, diff_list = [], [], []
    for _, r in cohort.iterrows():
        m = int(r['경과개월(선택일기준)'])
        if m <= 0:
            actual, plan, diff = np.nan, np.nan, np.nan
        else:
            actual = cum_rate(r, m)
            plan   = PLAN_loc.get(m, np.nan)
            diff   = (actual - plan) if pd.notna(actual) and pd.notna(plan) else np.nan
        actual_list.append(actual); plan_list.append(plan); diff_list.append(diff)

    cohort['실제누적(선택일)'] = actual_list
    cohort['계획누적(선택일)'] = plan_list
    cohort['편차(pp)'] = [(d*100 if pd.notna(d) else np.nan) for d in diff_list]
    cohort['실제누적세대(선택일)'] = (cohort['실제누적(선택일)'] * cohort['세대수']).round().astype('Int64')
    cohort['계획누적세대(선택일)'] = (cohort['계획누적(선택일)'] * cohort['세대수']).round().astype('Int64')
    cohort['현재_부족세대'] = (cohort['계획누적세대(선택일)'] - cohort['실제누적세대(선택일)']).clip(lower=0).astype('Int64')

    def needed_to_catch_up_next_month(row):
        m = int(row['경과개월(선택일기준)'])
        if m <= 0 or m >= MAX_M or pd.isna(row['실제누적(선택일)']):
            return np.nan
        next_plan = PLAN_loc.get(m+1, np.nan)
        if pd.isna(next_plan): return np.nan
        current_units = row['실제누적(선택일)'] * row['세대수']
        target_units  = next_plan * row['세대수']
        return max(0, target_units - current_units)

    cohort['다음달_계획맞춤_추가세대'] = cohort.apply(needed_to_catch_up_next_month, axis=1)

    out = cohort[cohort['편차(pp)'] < 0].copy().sort_values(by='편차(pp)', ascending=True)

    st.markdown(f"**계획 대비 저조 단지** (사용 데이터: 2025-01-01 ~ {end_date:%Y-%m-%d}, 세대수 ≥ {min_units}) — 상위 {top_n}개")
    st.dataframe(
        out.head(top_n)[[
            '아파트명','세대수','입주시작월','경과개월(선택일기준)',
            '실제누적세대(선택일)','계획누적세대(선택일)','현재_부족세대',
            '실제누적(선택일)','계획누적(선택일)','편차(pp)','다음달_계획맞춤_추가세대'
        ]].style.format({
            '실제누적(선택일)':'{:.1%}','계획누적(선택일)':'{:.1%}',
            '실제누적세대(선택일)':'{:,.0f}','계획누적세대(선택일)':'{:,.0f}',
            '현재_부족세대':'{:,.0f}','편차(pp)':'{:+.1f}','다음달_계획맞춤_추가세대':'{:,.0f}'
        }),
        use_container_width=True
    )

    # 막대 그래프
    worst = out.head(top_n)
    if not worst.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        labels = [f"{n} ({h}세대)" for n, h in zip(worst['아파트명'], worst['세대수'])]
        ax.barh(labels, worst['편차(pp)'])
        ax.set_xlabel('계획 대비 편차 (퍼센트포인트, 음수=저조)')
        ax.set_title('계획 대비 저조 단지 TOP')
        ax.invert_yaxis()
        for y, v in enumerate(worst['편차(pp)']):
            if pd.notna(v):
                ax.text(v + (1 if v < -1 else 0.2), y, f"{v:+.1f}pp", va='center')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # 버블 산포도
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sc = ax2.scatter(
            worst['계획누적(선택일)'], worst['실제누적(선택일)'],
            s=worst['세대수']*0.25, c=worst['편차(pp)'], cmap='coolwarm',
            alpha=0.9, edgecolors='k', linewidths=0.4
        )
        ax2.plot([0,1],[0,1],'--',linewidth=1)
        ax2.set_xlim(0,1); ax2.set_ylim(0,1)
        ax2.set_xlabel('계획 누적'); ax2.set_ylabel('실제 누적'); ax2.set_title('계획 vs 실제 (버블=세대수)')
        cb = plt.colorbar(sc); cb.set_label('편차(pp)')
        for _, r in worst.iterrows():
            if pd.notna(r['실제누적(선택일)']) and pd.notna(r['계획누적(선택일)']):
                ax2.text(r['계획누적(선택일)']+0.01, r['실제누적(선택일)']+0.01, str(r['아파트명'])[:8],
                         fontsize=9, alpha=0.85)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    return out

# =========================
# 5) 사이드바 입력 & 실행 버튼
# =========================
start_date_in = st.sidebar.date_input("시작일", date(2021, 1, 1))
end_date_in   = st.sidebar.date_input("종료일", date(2025, 12, 31))
min_units_in  = st.sidebar.number_input("세대수 하한(세대)", min_value=0, max_value=5000, value=300, step=50)
st.sidebar.markdown("---")

if st.sidebar.button("입주율 분석 실행", type="primary"):
    with st.spinner("분석 중..."):
        analyze_occupancy_by_period(pd.to_datetime(start_date_in), pd.to_datetime(end_date_in), min_units=min_units_in)
        st.divider()
        plot_yearly_avg_occupancy_with_plan(pd.to_datetime(start_date_in), pd.to_datetime(end_date_in), min_units=min_units_in)
        st.divider()
        recent2y_top_at_5m(pd.to_datetime(end_date_in), top_n=10, min_units=min_units_in)
        st.divider()
        cohort2025_progress(pd.to_datetime(end_date_in), min_units=min_units_in, MAX_M=9)
        st.divider()
        underperformers_vs_plan(pd.to_datetime(end_date_in), min_units=min_units_in, MAX_M=9, top_n=15)
else:
    st.info("좌측에서 **데이터/기간/세대수 하한**을 설정한 뒤 **입주율 분석 실행**을 눌러 주세요.")
