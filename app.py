# -*- coding: utf-8 -*-
import streamlit as st
import shap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import setup
from src.loaders.dataset import DatasetLoader
from src.loaders.model import ModelLoader
from src.constants.dataset import Dataset, Names
from src.process.feature import FeatureProcessor, NamesProcessor

MODEL_PATH = "./models/best_model.pkl"

# -------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------
st.set_page_config(
    page_title="Dashboard Executivo — Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# FUNÇÕES AUXILIARES
# -------------------------
def align_features(X: pd.DataFrame, model, fallback_names: list[str] | None = None):
    """Garante que as colunas estejam na mesma ordem do modelo."""
    if hasattr(model, "feature_names_in_"):
        needed = list(model.feature_names_in_)
    elif fallback_names:
        needed = fallback_names
    else:
        needed = list(X.columns)

    for col in needed:
        if col not in X.columns:
            X[col] = 0

    X = X[needed]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, needed

@st.cache_resource
def load_model_and_data():
    """Carrega modelo, dados e prepara para análise."""
    setup()
    model_loader = ModelLoader(MODEL_PATH)
    model = model_loader.load()
    dataset_loader = DatasetLoader()
    encoded_form_df = dataset_loader.load(Dataset.ENCODED_FORM)

    X, y, robust_feature = FeatureProcessor.process(encoded_form_df)
    X, used_feature_names = align_features(X, model, fallback_names=robust_feature)

    predictions = model.predict(X)
    prediction_proba = model.predict_proba(X)

    explainer = shap.TreeExplainer(model)
    shap_explain = explainer(X, check_additivity=False)

    shap_values_plot = shap_explain.values
    if shap_values_plot.ndim == 3:
        shap_values_plot = shap_values_plot[:, :, 1]

    robust_legible_names = NamesProcessor.process(used_feature_names)

    return model, X, y, predictions, prediction_proba, shap_values_plot, robust_legible_names

def calculate_financial_impact(num_students, cost_per_student=50000):
    """Calcula impacto financeiro da retenção de alunos."""
    return num_students * cost_per_student

# -------------------------
# DASHBOARD PRINCIPAL
# -------------------------
def main():
    st.title("🎓 Inteligência de Negócios — Evasão Escolar")
    st.markdown("""
    Este painel ajuda a entender **quando e por que alunos podem abandonar** e **quanto isso custa para a instituição**.  
    Use os controles à esquerda para ajustar cenários e ver os resultados em tempo real.
    """)
    st.divider()

    # Carrega dados
    with st.spinner("🔄 Carregando dados e modelo..."):
        model, X, y, predictions, prediction_proba, shap_values_plot, feature_names = load_model_and_data()

    # Sidebar — configurações
    st.sidebar.header("⚙️ Configurações de Cenário")
    cost_per_student = st.sidebar.number_input(
        "💰 Receita média anual por aluno (R$)",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=1000
    )
    risk_threshold = st.sidebar.slider(
        "🎯 Limite de probabilidade para risco de evasão",
        min_value=0.05, max_value=0.95,
        value=0.5, step=0.05
    )

    # KPIs gerais
    total_students = len(y)
    at_risk_mask = (prediction_proba[:, 1] >= risk_threshold)
    at_risk = int(at_risk_mask.sum())
    retention_rate = (total_students - at_risk) / total_students * 100
    potential_loss = calculate_financial_impact(at_risk, cost_per_student)

    st.markdown("### 📊 Visão Geral")
    st.markdown("Use esta visão para ter o panorama do risco e impacto financeiro.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Alunos Avaliados", f"{total_students:,}".replace(",", "."))
    col2.metric("⚠️ Em Risco de Evasão", f"{at_risk:,}".replace(",", "."))
    col3.metric("✅ Taxa Estimada de Retenção", f"{retention_rate:.1f}%")
    col4.metric("💸 Receita em Risco", f"R$ {potential_loss:,.0f}".replace(",", "."))

    st.divider()

    # Tabs para análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Causas da Evasão",
        "🏫 Perfil Institucional",
        "💡 Insights Interativos",
        "🎯 Cenários Financeiros",
        "📈 ROI & Simulação"
    ])

    # --------------------------
    # TAB 1 — CAUSAS DA EVASÃO
    with tab1:
        st.header("🔍 O que mais pesa no risco de evasão?")
        st.markdown("Escolha quantos fatores deseja visualizar abaixo:")

        top_n = st.slider("Número de fatores para mostrar", min_value=3, max_value=20, value=10, step=1)
        # calcula média abs SHAP por feature
        mean_abs_shap = np.mean(np.abs(shap_values_plot), axis=0)
        df_shap = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False).head(top_n)

        fig = px.bar(df_shap, x="importance", y="feature", orientation="h",
                     labels={"importance":"Impacto médio", "feature":"Fator"},
                     title=f"Top {top_n} fatores de risco (orientação horizontal)")
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Interpretação rápida:** Um fator mais alto significa que ele tem maior influência em aumentar risco de evasão.")

    # --------------------------
    # TAB 2 — PERFIL INSTITUCIONAL
    with tab2:
        st.header("🏫 Como diferentes perfis se comportam?")
        st.markdown("Filtre por dimensão para ver o impacto:")
        categories = {
            "Socioeconômica": ["Situação de Moradia", "Trabalho Atual", "Bolsa de Estudos"],
            "Geográfica": ["Cidade de Origem", "Frequência de Retorno", "Natural de SRS"],
            "Acadêmica": ["Dependências", "Período Atual", "Tipo de Escola"],
            "Comportamental": ["Horas de Estudo", "Abandono por Trabalho", "Atividades Extracurriculares", "Trancamento Anterior", "Evasão Anterior"],
            "Demográfica": ["Faixa de Idade", "Gênero"]
        }
        selected_cat = st.selectbox("Selecionar dimensão", list(categories.keys()))
        feats = categories[selected_cat]
        idx = [i for i, name in enumerate(feature_names) if name in feats]
        if idx:
            vals = [mean_abs_shap[i] for i in idx]
            fig2 = px.pie(
                names=[feature_names[i] for i in idx],
                values=vals,
                title=f"Distribuição de impacto — {selected_cat}"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f"Na dimensão **{selected_cat}**, os fatores analisados estão representados. Use essa visão para entender onde sua instituição pode priorizar.")
        else:
            st.info("Nenhum fator disponível para essa dimensão.")

    # --------------------------
    # TAB 3 — INSIGHTS INTERATIVOS
    with tab3:
        st.header("💡 Faça cenários – filtre e compare")
        st.markdown("Utilize os filtros abaixo para explorar diferentes condições:")

        # Exemplo: filtro por “Bolsa de Estudos” valor binário
        filter_bolsa = st.selectbox("Mostrar apenas alunos com Bolsa de Estudos?", ["Todos", "Sim", "Não"])
        df_full = X.copy()
        # localiza coluna “Bolsa de Estudos”
        col_bolsa = [c for c in X.columns if "Bolsa de Estudos" in Names.LEGIBLE_NAMES.get(c, "") or "Bolsa" in Names.LEGIBLE_NAMES.get(c, "")]
        # se existir
        if col_bolsa:
            colb = col_bolsa[0]
            if filter_bolsa == "Sim":
                df_full = df_full[df_full[colb] == 1]
            elif filter_bolsa == "Não":
                df_full = df_full[df_full[colb] == 0]

        st.write(f"Alunos considerados no filtro: {len(df_full):,}".replace(",", "."))

        # mostra gráfico de risco médio por filtro
        avg_proba = prediction_proba[df_full.index, 1].mean()
        st.metric("Probabilidade média de evasão (%)", f"{avg_proba*100:.1f}%")

        fig3 = px.histogram(
            prediction_proba[:, 1],
            nbins=30,
            title="Distribuição das probabilidades de evasão",
            labels={"value":"Probabilidade de evasão", "count":"Número de alunos"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --------------------------
    # TAB 4 — CENÁRIOS FINANCEIROS
    with tab4:
        st.header("🎯 Cenários de Impacto financeiro rápido")
        st.markdown("Arraste os controles e veja como a receita muda:")

        scenario_students = st.slider("Número de alunos em risco no cenário", 0, total_students, at_risk, step=10)
        scenario_cost = st.number_input("Valor por aluno (R$)", min_value=10000, max_value=200000, value=cost_per_student, step=5000)

        success_rates = [0, 25, 50, 75, 100]
        impacts = [calculate_financial_impact(int(scenario_students * r / 100), scenario_cost) for r in success_rates]
        df_imp = pd.DataFrame({
            "Taxa de Retenção (%)": success_rates,
            "Receita Preservada (R$)": [i for i in impacts]
        })
        fig4 = px.line(df_imp, x="Taxa de Retenção (%)", y="Receita Preservada (R$)",
                       title="Receita preservada por taxa de retenção",
                       markers=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)

    # --------------------------
    # TAB 5 — ROI & SIMULAÇÃO
    with tab5:
        st.header("📈 ROI & Planejamento")
        st.markdown("Configure seu investimento e verifique o retorno esperado:")

        col1, col2, col3 = st.columns(3)
        success_rate = col1.slider("Taxa de sucesso (%)", 10, 90, 50)
        intervention_cost = col2.number_input("Custo por aluno (R$)", 100, 10000, 2000)
        total_invest = col3.number_input("Investimento total (R$)", 10000, 2000000, 500000)

        expected_saved = int((success_rate / 100) * at_risk)
        total_saved = calculate_financial_impact(expected_saved, cost_per_student)
        roi = ((total_saved - total_invest) / total_invest) * 100 if total_invest else 0

        st.metric("👥 Alunos Retidos Esperados", f"{expected_saved:,}".replace(",", "."))
        st.metric("💵 Receita Preservada", f"R$ {total_saved:,.0f}".replace(",", "."))
        st.metric("📊 ROI Estimado", f"{roi:.1f}%")

        st.markdown("""
        Use esta simulação para **planejar orçamento de retenção** e justificar investimentos com impacto estimado.
        """)

    # --------------------------
    # RODAPÉ
    st.divider()
    st.markdown("""
    <hr style="margin-top:1em;margin-bottom:1em;">
    <div style='text-align:center;color:gray;font-size:0.9em;'>
        Desenvolvido por <b>Equipe de Análise Preditiva</b> ·  
        Baseado em dados acadêmicos e socioeconômicos.<br>
        © 2025 — Sistema de Inteligência Institucional
    </div>
    """, unsafe_allow_html=True)

if __name__:
    main()
