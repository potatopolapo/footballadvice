import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 페이지 기본 설정 (탭 제목 + 아이콘)
# -----------------------------
st.set_page_config(
    page_title="선수 유형 나누기",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ 선수 유형 나누기")
st.caption("EAFC25 상위 100명 선수 데이터를 이용한 K-평균 군집 분석")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/eafc25_top100.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 능력치 영문 -> 한글 이름 매핑
STAT_MAP = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움"
}
STAT_COLS_EN = list(STAT_MAP.keys())
STAT_COLS_KO = list(STAT_MAP.values())

# 한글 이름 컬럼을 추가로 만들어서 사용 (원본은 그대로 두고 보기용 컬럼 생성)
df_display = df.rename(columns=STAT_MAP)

# -----------------------------
# 사용자 입력: 묶는 데 사용할 능력치 선택
# -----------------------------
st.subheader("1. 묶음(군집)에 사용할 능력치 선택")

selected_stats_ko = st.multiselect(
    "군집화에 사용할 능력치를 2개 이상 선택하세요.",
    options=STAT_COLS_KO,
    default=STAT_COLS_KO  # 기본값: 여섯 개 다
)

if len(selected_stats_ko) < 2:
    st.warning("능력치를 2개 이상 선택해야 분석을 진행할 수 있습니다.")
    st.stop()

# 한글 -> 영문 역매핑 (선택된 능력치를 원본 컬럼명으로 변환)
KO_TO_EN = {v: k for k, v in STAT_MAP.items()}
selected_stats_en = [KO_TO_EN[s] for s in selected_stats_ko]

# -----------------------------
# 사용자 입력: 묶음 수(k) 선택
# -----------------------------
k = st.slider("묶음(군집) 개수를 선택하세요.", min_value=2, max_value=6, value=3, step=1)

# -----------------------------
# 표준화 (선택된 능력치 기준)
# -----------------------------
X = df[selected_stats_en].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-평균 군집화 (현재 선택된 k로 실제 분석에 사용)
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)

df_display["cluster_raw"] = raw_labels

# -----------------------------
# 슈팅 평균이 큰 묶음부터 ㉮, ㉯, ㉰... 순서로 이름 붙이기
# -----------------------------
cluster_order = (
    df_display.groupby("cluster_raw")["슈팅"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# 원 안의 한글 순서 문자 목록 (최대 6개까지 지원)
CIRCLE_LETTERS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]

label_map = {raw: CIRCLE_LETTERS[i] for i, raw in enumerate(cluster_order)}
df_display["묶음"] = df_display["cluster_raw"].map(label_map)

# 묶음 순서를 ㉮, ㉯, ㉰... 순서대로 정렬되게 카테고리 지정
ordered_labels = [label_map[raw] for raw in cluster_order]
df_display["묶음"] = pd.Categorical(df_display["묶음"], categories=ordered_labels, ordered=True)

# -----------------------------
# 2차원 산점도
# -----------------------------
st.subheader("2. 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox("가로축(X) 능력치", STAT_COLS_KO, index=0, key="x2d")
with col2:
    y_axis_2d = st.selectbox("세로축(Y) 능력치", STAT_COLS_KO, index=1, key="y2d")

fig_2d = px.scatter(
    df_display,
    x=x_axis_2d,
    y=y_axis_2d,
    color="묶음",
    hover_name="name_ko",
    category_orders={"묶음": ordered_labels},
    title=f"{x_axis_2d} vs {y_axis_2d} 2차원 산점도"
)
fig_2d.update_traces(marker=dict(size=8))
st.plotly_chart(fig_2d, use_container_width=True)

# -----------------------------
# 3차원 산점도
# -----------------------------
st.subheader("3. 3차원 산점도")

if len(selected_stats_ko) < 3:
    st.info("선택한 능력치가 3개 미만이라 3차원 산점도를 그릴 수 없습니다. 능력치를 3개 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox("X축 능력치", STAT_COLS_KO, index=0, key="x3d")
    with col4:
        y_axis_3d = st.selectbox("Y축 능력치", STAT_COLS_KO, index=1, key="y3d")
    with col5:
        z_axis_3d = st.selectbox("Z축 능력치", STAT_COLS_KO, index=2, key="z3d")

    fig_3d = px.scatter_3d(
        df_display,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="묶음",
        hover_name="name_ko",
        category_orders={"묶음": ordered_labels},
        title=f"{x_axis_3d} · {y_axis_3d} · {z_axis_3d} 3차원 산점도"
    )
    # 점 크기를 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------
# 묶음별 인원 수 + 여섯 능력치 평균 표
# -----------------------------
st.subheader("4. 묶음별 인원 및 능력치 평균")

summary_table = (
    df_display.groupby("묶음", observed=True)[STAT_COLS_KO]
    .mean()
    .round(2)
)
count_table = df_display.groupby("묶음", observed=True).size().rename("인원 수")

final_summary = pd.concat([count_table, summary_table], axis=1)
st.dataframe(final_summary, use_container_width=True)

# -----------------------------
# 묶음별 종합 능력치(overall) 상위 5명
# -----------------------------
st.subheader("5. 묶음별 종합 능력치(overall) 상위 5명")

for label in ordered_labels:
    st.markdown(f"**{label} 묶음**")
    top5 = (
        df_display[df_display["묶음"] == label]
        .sort_values("overall", ascending=False)
        .head(5)[["name_ko", "overall"]]
        .reset_index(drop=True)
    )
    top5.index = top5.index + 1  # 1등, 2등... 보기 좋게
    st.table(top5)

# -----------------------------
# 포지션 분류 + 묶음-포지션 교차표
# -----------------------------
st.subheader("6. 묶음과 포지션 교차표")

def classify_position(positions_str):
    """positions 열의 맨 앞 포지션을 보고 공격수/미드필더/수비수로 분류"""
    first_pos = str(positions_str).split(",")[0].strip()

    forward_positions = ["ST", "CF", "LW", "RW"]
    midfielder_positions = ["CAM", "CM", "CDM", "LM", "RM"]
    defender_positions = ["CB", "LB", "RB", "LWB", "RWB"]

    if first_pos in forward_positions:
        return "공격수"
    elif first_pos in midfielder_positions:
        return "미드필더"
    elif first_pos in defender_positions:
        return "수비수"
    else:
        return "기타"

df_display["포지션군"] = df_display["positions"].apply(classify_position)

cross_tab = pd.crosstab(df_display["묶음"], df_display["포지션군"])
# 묶음 순서대로 재정렬
cross_tab = cross_tab.reindex(ordered_labels)

st.dataframe(cross_tab, use_container_width=True)

st.caption("※ 포지션 정보는 군집화(묶음 나누기)에 사용되지 않았으며, 결과 확인용으로만 사용되었습니다.")

# -----------------------------
# 7. 묶음 수를 정하는 화면 (엘보우 방법 + 실루엣 점수)
# -----------------------------
st.subheader("7. 적절한 묶음 수 정하기")

st.markdown("""
현재 선택한 능력치를 기준으로, 묶음 수를 1개부터 7개까지 바꿔가며
**관성(inertia, 각 점이 자기 묶음 중심에서 떨어진 거리의 제곱을 모두 더한 값)** 을 계산합니다.

이 값은 묶음 수가 늘어날수록 항상 줄어들지만, 어느 순간부터는 줄어드는 정도가 작아집니다.
이렇게 꺾이는 지점(팔꿈치, elbow)을 찾아 적절한 묶음 수를 정하는 방법을 **엘보우 방법**이라고 합니다.
""")

# 묶음 수 1~7에 대해 관성 계산
k_range = list(range(1, 8))
inertia_list = []

for k_candidate in k_range:
    km_temp = KMeans(n_clusters=k_candidate, random_state=42, n_init=10)
    km_temp.fit(X_scaled)
    inertia_list.append(km_temp.inertia_)

inertia_df = pd.DataFrame({
    "묶음 수": k_range,
    "관성(inertia)": inertia_list
})

# 꺾은선 그래프 그리기
fig_elbow = px.line(
    inertia_df,
    x="묶음 수",
    y="관성(inertia)",
    markers=True,
    title="묶음 수에 따른 관성(inertia) 변화 (엘보우 방법)"
)

# 현재 고른 묶음 수 위치에 세로선 긋기
fig_elbow.add_vline(
    x=k,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {k}개",
    annotation_position="top right"
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 묶음 수마다 관성 값과 바로 앞 값 대비 감소량을 표로 보여주기
inertia_df["직전 대비 감소량"] = inertia_df["관성(inertia)"].shift(1) - inertia_df["관성(inertia)"]
inertia_df_display = inertia_df.copy()
inertia_df_display["관성(inertia)"] = inertia_df_display["관성(inertia)"].round(2)
inertia_df_display["직전 대비 감소량"] = inertia_df_display["직전 대비 감소량"].round(2)

st.markdown("**묶음 수별 관성 값과 직전 대비 감소량**")
st.dataframe(inertia_df_display, use_container_width=True, hide_index=True)

# -----------------------------
# 8. 실루엣 점수 (묶음 수 2~7)
# -----------------------------
st.subheader("8. 실루엣 점수로 묶음 수 살펴보기")

st.markdown("""
**실루엣 점수**는 각 점이 자기가 속한 묶음 안에서는 얼마나 가깝게 모여 있고,
다른 묶음과는 얼마나 멀리 떨어져 있는지를 나타내는 값입니다.
1에 가까울수록 묶음이 잘 나뉘어 있다는 뜻이고, 0에 가까우면 묶음 구분이 애매하다는 뜻입니다.
(묶음이 1개일 때는 계산할 수 없어서 2개부터 계산합니다.)
""")

silhouette_k_range = list(range(2, 8))
silhouette_scores = []

for k_candidate in silhouette_k_range:
    km_temp = KMeans(n_clusters=k_candidate, random_state=42, n_init=10)
    labels_temp = km_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    silhouette_scores.append(score)

silhouette_df = pd.DataFrame({
    "묶음 수": silhouette_k_range,
    "실루엣 점수": silhouette_scores
})

fig_silhouette = px.line(
    silhouette_df,
    x="묶음 수",
    y="실루엣 점수",
    markers=True,
    title="묶음 수에 따른 실루엣 점수 변화"
)

fig_silhouette.add_vline(
    x=k,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택: {k}개",
    annotation_position="top right"
)

st.plotly_chart(fig_silhouette, use_container_width=True)

silhouette_df_display = silhouette_df.copy()
silhouette_df_display["실루엣 점수"] = silhouette_df_display["실루엣 점수"].round(4)

st.markdown("**묶음 수별 실루엣 점수**")
st.dataframe(silhouette_df_display, use_container_width=True, hide_index=True)
