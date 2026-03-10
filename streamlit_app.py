
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(layout="wide")

st.title("📊 Dashboard Educacional - Passos Mágicos")

# =========================
# CARREGAR DADOS
# =========================

@st.cache_data
def load_data():
    df = pd.read_excel("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
    return df

df = load_data()

# =========================
# LIMPEZA
# =========================

df.columns = df.columns.str.strip()

numeric_cols = [
"IAA","IEG","IPS","IPP","IDA","IPV","IAN","INDE 22","INDE 23","INDE 2023","INDE 2024"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# VISÃO GERAL
# =========================

st.header("Visão Geral")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total alunos",len(df))

if "INDE 2024" in df.columns:
    col2.metric("INDE médio",round(df["INDE 2024"].mean(),2))

col3.metric("IEG médio",round(df["IEG"].mean(),2))
col4.metric("IDA médio",round(df["IDA"].mean(),2))

# =========================
# 1 ADEQUAÇÃO NÍVEL
# =========================

st.header("1️⃣ Adequação do nível (IAN)")

fig = px.histogram(df,x="IAN",title="Distribuição do Indicador de Adequação ao Nível")

st.plotly_chart(fig,use_container_width=True)

# =========================
# 2 DESEMPENHO ACADÊMICO
# =========================

st.header("2️⃣ Desempenho Acadêmico (IDA)")

fig = px.box(
df,
y="IDA",
title="Distribuição do desempenho acadêmico"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 3 ENGAJAMENTO
# =========================

st.header("3️⃣ Engajamento vs Desempenho")

fig = px.scatter(
df,
x="IEG",
y="IDA",
color="Pedra",
trendline="ols"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 4 AUTOAVALIAÇÃO
# =========================

st.header("4️⃣ Autoavaliação vs desempenho")

fig = px.scatter(
df,
x="IAA",
y="IDA",
color="Pedra",
trendline="ols"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 5 PSICOSSOCIAL
# =========================

st.header("5️⃣ Aspectos psicossociais")

fig = px.scatter(
df,
x="IPS",
y="IDA",
color="Pedra",
trendline="ols"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 6 PSICOPEDAGÓGICO
# =========================

st.header("6️⃣ Avaliação Psicopedagógica")

fig = px.scatter(
df,
x="IPP",
y="IAN",
color="Pedra",
trendline="ols"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 7 PONTO DE VIRADA
# =========================

st.header("7️⃣ Ponto de Virada")

fig = px.scatter(
df,
x="IPV",
y="IDA",
color="Pedra"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 8 MULTIDIMENSIONALIDADE
# =========================

st.header("8️⃣ Relação entre indicadores")

fig = px.scatter_matrix(
df,
dimensions=["IDA","IEG","IPS","IPP","IPV"],
color="Pedra"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 9 MACHINE LEARNING
# =========================

st.header("9️⃣ Previsão de risco")

df_ml = df.dropna(subset=["IAN","IEG","IPS","IDA","IPP","IPV"])

df_ml["risco"] = (
(df_ml["IAN"] < df_ml["IAN"].quantile(0.25))
).astype(int)

features = ["IEG","IPS","IPP","IDA","IPV"]

X = df_ml[features]
y = df_ml["risco"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.3,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

st.metric("Acurácia modelo",round(acc,2))

report = classification_report(y_test,pred)

st.text(report)

# importância das variáveis

importance = pd.DataFrame({
"variavel":features,
"importancia":model.feature_importances_
})

fig = px.bar(
importance,
x="variavel",
y="importancia",
title="Importância das variáveis na previsão de risco"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 10 EFETIVIDADE PROGRAMA
# =========================

st.header("🔟 Distribuição das Pedras")

fig = px.histogram(
df,
x="Pedra",
title="Classificação educacional dos alunos"
)

st.plotly_chart(fig,use_container_width=True)

# =========================
# 11 INSIGHTS
# =========================

st.header("💡 Insights")

corr = df[["IEG","IDA"]].corr().iloc[0,1]

st.write("Correlação Engajamento vs Desempenho:",round(corr,2))

corr2 = df[["IPS","IDA"]].corr().iloc[0,1]

st.write("Correlação Psicossocial vs Desempenho:",round(corr2,2))

st.write("Média INDE:",round(df["INDE 2024"].mean(),2))