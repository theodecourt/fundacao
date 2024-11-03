import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import streamlit as st
import plotly.express as px

num_romanos = {1: 'I', 2: 'II', 3: 'III'}

def criar_tabela_exemplo(idioma):
    return pd.DataFrame({
        "Carga (tf)": [1200, 1125, 1050, 975, 900, 825, 750, 675, 600, 525, 450, 375, 300, 225, 150, 75],
        "Recalque (mm)": [27.21, 24.55, 21.95, 19.35, 17.28, 14.72, 12.81, 11.03, 9.52, 8.30, 6.92, 5.19, 3.79, 2.48, 1.51, 0.66]
    })

def carregar_tabela(idioma):
    uploaded_file = st.file_uploader(
        "Escolha o arquivo CSV ou XLSX" if idioma == "Português" else "Choose the CSV or XLSX file", 
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        if 'csv' in uploaded_file.name:
            return pd.read_csv(uploaded_file, delimiter=';')
        else:
            return pd.read_excel(uploaded_file)
    else:
        st.write("Por favor, carregue um arquivo para continuar.")
    return None

def calcular_regressao(tabela, num_regressoes, pontos_tipos, idioma):
    regressions = []
    tipos = []
    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_regressao = pontos_tipos[i]
        linear = tabela[lin_in-1:lin_fim]
        
        if tipo_regressao == 'linear':
            reg = np.polyfit(linear['Carga'], linear['rigidez'], deg=1)
        else:  # log
            reg = np.polyfit(linear['logQ'], linear['logRig'], deg=1)
        
        regressions.append(reg)
        tipos.append(tipo_regressao)
    
    return regressions, tipos

def realizar_calculos_independentes(recalque_input, carga_input, regressions, tipos):
    for i, (reg, tipo_regressao) in enumerate(zip(regressions, tipos)):
        if recalque_input > 0:
            if tipo_regressao == 'linear':
                carga_calculada = recalque_input * reg[0] + reg[1]
            else:
                carga_calculada = 10 ** (reg[0] * np.log10(recalque_input) + reg[1])
            st.write(f"Para a Regressão {num_romanos[i+1]}, dado o recalque {recalque_input:.2f} mm, a carga é {carga_calculada:.2f} tf.")

        if carga_input > 0:
            if tipo_regressao == 'linear':
                recalque_calculado = (carga_input - reg[1]) / reg[0]
            else:
                recalque_calculado = 10 ** ((np.log10(carga_input) - reg[1]) / reg[0])
            st.write(f"Para a Regressão {num_romanos[i+1]}, dada a carga {carga_input:.2f} tf, o recalque é {recalque_calculado:.2f} mm.")

def primeiro_programa(idioma):
    st.title("Análise de Carga e Recalque")
    
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        st.write("Dados de Carga e Recalque:")
        st.dataframe(tabela)

        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ'] = np.log10(tabela['Carga'])
        tabela['logRig'] = np.log10(tabela['rigidez'])

        fig = px.scatter(tabela, x="Carga", y="Recalque", title="Carga vs Recalque")
        st.plotly_chart(fig)

        fig2 = px.scatter(tabela, x="Carga", y="rigidez", title="Carga vs Rigidez")
        st.plotly_chart(fig2)

        num_regressoes = st.selectbox('Quantas regressões:', [1, 2, 3], index=0)
        
        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in = st.number_input(f'Ponto inicial da regressão {num_romanos[i+1]}:', min_value=1, max_value=len(tabela), value=1)
            lin_fim = st.number_input(f'Ponto final da regressão {num_romanos[i+1]}:', min_value=lin_in, max_value=len(tabela), value=len(tabela))
            tipo_regressao = st.selectbox(f'Tipo de regressão {num_romanos[i+1]}:', ['linear', 'log'], index=0)
            pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
        
        regressions, tipos = calcular_regressao(tabela, num_regressoes, pontos_tipos, idioma)

        recalque_input = st.number_input('Informe o recalque para cálculo independente (mm):', format="%.2f", step=0.01)
        carga_input = st.number_input('Informe a carga para cálculo independente (tf):', format="%.2f", step=0.01)

        realizar_calculos_independentes(recalque_input, carga_input, regressions, tipos)

idioma = 'Português'
primeiro_programa(idioma)


