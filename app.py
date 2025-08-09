# ===========================
# app.py  (Streamlit 대시보드)
# ===========================
import os, logging, warnings
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
from datetime import datetime

# ---------------------------------------------------
# 0) 한글 폰트 적용 (프로젝트 내 /fonts/NanumGothic-Regular.ttf 사용)
# ---------------------------------------------------
def set_korean_font():
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic-Regular.ttf")
    chosen = None
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            chosen = prop.get_name()
            mpl.rcParams['font.family'] = chosen
            mpl.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        except Exception:
            pass
    mpl.rcParams['axes.unicode_minus'] = False
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    return chosen or "시스템 기본"

applied_font = set_korean_font()

# ---------------------------------------------------
# 1) 페이지 기본 설정
# ---------------------------------------------------
st.set_page_config(page_title="입주율 분석 대시보드", layout="wide")
st.markdown(f"**✅ 적용된 한글 폰트:** {applied_font}")

# ---------------------------------------------------
# 2) 데이터 업로드
#   - 엑셀 파일 내 시트명은 반드시 'data'
#   - 주요 컬럼: 아파트명, 공급승인일자, 세대수, 그리고 '1개월'...'n개월' 컬럼들
# ---------------------------------------------------
st.sidebar.header("데이터 업로드")
uploaded_file = st.sidebar.file_uploader("입주율.xlsx 업로드 (시트명: data)", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_excel(file, sheet_name="data")
    df['공급승인일자'] = pd.to_datetime(df['공급승인일자'], errors='coerce')
    return df

if uploaded_file is None:
    st.info("좌측에서 **입주율.xlsx** 파일을 업로드하면 시작됩니다. (시트명: `data`)")
    st.stop()

df = load_data(uploaded_file)

# ---------------------------------------------------
# 3) 공통 유틸: 입주시작 index, 입주시작월 계산 및 월열 정렬
# ---------------------------------------------------
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
        if pd.notna(r['입주시작index']) and pd.notna(r['공급승인일자']) else pd.NaT, axis=1
    )
    return month_cols

month_cols = ensure_start_index(df)

# ---------------------------------------------------
# 4) 사이드바 옵션
# ---------------------------------------------------
min_date = pd.to_datetime(df['공급승인일자'].min()) if pd.notna(df['공급승인일자']).any() else pd.Timestamp("2020-01-01")
max_date = pd.Timestamp.today()

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("시작일", value=min_date.date())
end_date   = c2.date_input("종료일", value=max_date.date())
min_units  = st.sidebar.slider("세대수 하한(세대)", min_value=0, max_value=2000, value=300, step=50)
top_n      = st.sidebar.slider("TOP N (5개월차)", min_value=5, max_value=20, value=10, step=1)

start_date = pd.to_datetime(start_date)
end_date   = pd.to_datetime(end_date)

# ---------------------------------------------------
# 5) 분석 1: 기간 요약 + 상위 10
# ---------------------------------------------------
st.subheader("① 기간 요약 + 상위 10")
def analyze_occupancy_by_period(시작일, 종료일, min_units=0):
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
        lambda r: max(0, min(len(month_cols)-1, (종료일.year - r['공급승인일자'].year)*12 + (종료일.month - r['공급승인일자'].month)) - int(r['입주시작index']) + 1)
        if pd.notna(r['입주시작index']) else np.nan, axis=1
    )
    base['입주율'] = base['입주세대수'] / base['세대수']

    result_df = base[['아파트명','공급승인일자','세대수','입주시작월','입주세대수','입주기간(개월)','입주율']]\
                  .dropna(subset=['입주세대수']).sort_values(by='입주율', ascending=False)

    st.markdown(f"**기간:** {시작일:%Y-%m-%d} ~ {종료일:%Y-%m-%d}  |  **세대수 하한:** {min_units}세대")
    st.dataframe(result_df.reset_index(drop=True).style.format({'입주율':'{:.1%}'}), use_container_width=True)

    top = result_df.head(10)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top['아파트명'], top['입주율'])
        ax.set_xlabel('입주율'); ax.set_title('입주율 상위 10개 단지')
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(top['입주율']):
            ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va='center')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("그래프를 그릴 데이터가 없습니다.")
    return result_df

_ = analyze_occupancy_by_period(start_date, end_date, min_units=min_units)

# ---------------------------------------------------
# 6) 분석 2: 연도별 입주시작 단지의 월별 누적 입주율
# ---------------------------------------------------
st.subheader("② 연도별 입주시작 단지의 월별 누적 입주율")
def plot_yearly_avg_occupancy_with_plan(start_date, end_date, min_units=0):
    MAX_M = 9
    cohort = df[
        (df['입주시작월'] >= start_date) &
        (df['입주시작월'] <= end_date) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()
    cohort['입주시작연도'] = cohort['입주시작월'].dt.year

    rate_dict = {}
    has_data = False
    fig, ax_plot = plt.subplots(figsize=(12, 6))
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

    # 사업계획 기준선 (퍼센트 → 비율)
    계획입주율 = {1:9.29, 2:43.25, 3:62.75, 4:72.61, 5:78.17, 6:81.56, 7:84.28, 8:86.07, 9:87.86}
    plan_x = list(range(1, MAX_M+1)); plan_y = [계획입주율[i]/100 for i in plan_x]
    ax_plot.plot(plan_x, plan_y, linestyle='--', marker='x', label='사업계획 기준')

    if has_data:
        ax_plot.set_title(f"연도별 입주시작 단지의 월별 누적 입주율 (세대수 ≥ {min_units})")
        ax_plot.set_xlabel("입주경과 개월")
        ax_plot.set_ylabel("누적 평균 입주율")
        ax_plot.set_xticks(plan_x); ax_plot.set_xticklabels([f"{i}개월" for i in plan_x])
        ax_plot.set_ylim(0, 1); ax_plot.grid(True); ax_plot.legend(ncol=3, loc='lower right')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("표시할 연도별 입주율 데이터가 없습니다.")

plot_yearly_avg_occupancy_with_plan(start_date, end_date, min_units=min_units)

# ---------------------------------------------------
# 7) 분석 3: 최근 2년 — 5개월차 입주율 TOP
# ---------------------------------------------------
st.subheader("③ 최근 2년 — 5개월차 입주율 TOP")
def recent2y_top_at_5m(end_date, top_n=10, min_units=0):
    start_cal = pd.Timestamp(year=end_date.year - 1, month=1, day=1)
    cohort = df[
        (df['입주시작월'] >= start_cal) &
        (df['입주시작월'] <= end_date) &
        (df['입주시작index'].notna()) &
        (df['세대수'].notna()) &
        (df['세대수'] >= min_units)
    ].copy()
    eligible = cohort[(cohort['입주시작월'] + pd.offsets.DateOffset(months=4)) <= end_date].copy()
    if eligible.empty:
        st.warning("최근 2년 코호트에서 5개월차까지 도달한 단지가 없습니다.")
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
        '입주율_3개월': '{:.1%}', '입주율_4개월': '{:.1%}', '입주율_5개월': '{:.1%}'
    }), use_container_width=True)

    top = ranked.head(top_n).copy()
    if not top.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f'{n} ({h}세대)' for n, h in zip(top['아파트명'], top['세대수'])]
        ax.barh(labels, top['입주율_5개월'])
        ax.set_xlabel('입주시작 5개월차 입주율')
        ax.set_title(f'최근 2년 — 5개월차 입주율 TOP (세대수 ≥ {min_units})')
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        for y, v in enumerate(top['입주율_5개월']):
            ax.text(min(v + 0.01, 0.98), y, f"{v*100:.1f}%", va='center')
        fig.tight_layout()
        st.pyplot(fig)

    return ranked

_ = recent2y_top_at_5m(end_date, top_n=top_n, min_units=min_units)

st.caption("© 입주율 분석 대시보드")
