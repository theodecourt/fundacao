import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve

# Função para carregar a tabela
def carregar_tabela():
    uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, delimiter=';')

# Função para calcular a interseccao entre duas regressões
def calcular_interseccao(reg1, reg2, tipo1, tipo2):
    if tipo1 == 'linear' and tipo2 == 'linear':
        A = np.array([[reg1[0], -1], [reg2[0], -1]])
        B = np.array([-reg1[1], -reg2[1]])
        interseccao = np.linalg.solve(A, B)
    elif tipo1 == 'log' and tipo2 == 'log':
        A = np.array([[reg1[0], -1], [reg2[0], -1]])
        B = np.array([-reg1[1], -reg2[1]])
        interseccao_log = np.linalg.solve(A, B)
        interseccao = 10**interseccao_log
    elif tipo1 == 'linear' and tipo2 == 'log':
        def func_intersec(x):
            return reg1[0] * x + reg1[1] - 10**(reg2[0] * np.log10(x) + reg2[1])
        interseccao_carga = fsolve(func_intersec, x0=1)
        interseccao_rigidez = reg1[0] * interseccao_carga + reg1[1]
        interseccao = [interseccao_carga[0], interseccao_rigidez[0]]
    elif tipo1 == 'log' and tipo2 == 'linear':
        def func_intersec(x):
            return 10**(reg1[0] * np.log10(x) + reg1[1]) - reg2[0] * x - reg2[1]
        interseccao_carga = fsolve(func_intersec, x0=1)
        interseccao_rigidez = reg2[0] * interseccao_carga + reg2[1]
        interseccao = [interseccao_carga[0], interseccao_rigidez[0]]
    return interseccao

# Função para calcular a regressão e plotar os gráficos
def calcular_regressao(tabela, num_regressoes, pontos_tipos):
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    
    colors = ['b', 'red', 'green']
    plt.plot(x0, y0, 'go', label='Dados Originais')
    
    regressions = []
    tipos = []
    
    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_regressao = pontos_tipos[i]
        linear = tabela[lin_in-1:lin_fim]
        if tipo_regressao == 'linear':
            reg = np.polyfit(linear['Carga'], linear['rigidez'], deg=1)
            predict = np.poly1d(reg)
            x = np.linspace(0, tabela['Carga'].max(), 100)
            y = predict(x)
            corr_matrix = np.corrcoef(linear['rigidez'], linear['Carga'])
            equacao = f'rigidez = {reg[0]:.4f} * Carga + {reg[1]:.4f}'
        else:  # log
            reg = np.polyfit(linear['logQ'], linear['logRig'], deg=1)
            predict = np.poly1d(reg)
            x = np.linspace(0, tabela['Carga'].max(), 100)
            y = 10**predict(np.log10(x))
            corr_matrix = np.corrcoef(linear['logRig'], linear['logQ'])
            equacao = f'log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}'
        
        corr = corr_matrix[0, 1]
        R_sq = corr**2

        plt.plot(x, y, colors[i], label=f'Regressão {i+1}')
        
        regressions.append(reg)
        tipos.append(tipo_regressao)
        
        st.write(f'Pontos utilizados na regressão {i+1}: ', lin_in, ' até ', lin_fim)
        st.write('Tipo de regressão: ', tipo_regressao.capitalize())
        st.write('Equação da regressão: ', equacao)
        st.write('R2: ', R_sq)
    
    # Calcular e mostrar pontos de interseccao entre todas as combinações possíveis
    for i in range(num_regressoes):
        for j in range(i + 1, num_regressoes):
            interseccao = calcular_interseccao(regressions[i], regressions[j], tipos[i], tipos[j])
            plt.plot(interseccao[0], interseccao[1], 'rx')  # Marca a interseccao com um 'x' vermelho
            st.write(f'Interseccao entre regressão {i+1} e {j+1}: Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}')
    
    plt.xlabel('Carga')
    plt.ylabel('Rigidez')
    plt.title('Regressão de Carga x Rigidez')
    plt.legend().set_visible(False)
    st.pyplot(plt)

# Função para a página "Programas"
def pagina_programas():
    st.title('Programas de Luciano Decourt')

    tabela = carregar_tabela()
    if tabela is not None:
        # Plota os gráficos antes de exibir as opções de regressões
        fig = px.scatter(tabela, x="Carga", y="Recalque")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)

        tabela['rigidez'] = tabela.apply(lambda row: row.Carga / row.Recalque, axis=1)
        fig2 = px.scatter(tabela, x="Carga", y="rigidez")
        st.plotly_chart(fig2)

        tabela['logQ'] = tabela.apply(lambda row: math.log(row.Carga, 10), axis=1)
        tabela['logReq'] = tabela.apply(lambda row: math.log(row.Recalque, 10), axis=1)
        tabela['logRig'] = tabela.apply(lambda row: math.log(row.rigidez, 10), axis=1)
        
        # Escolha o número de regressões
        num_regressoes = st.selectbox('Quantas regressões:', [1, 2, 3], index=0)
        
        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in = st.number_input(f'Ponto inicial {i+1}:', min_value=1, value=1)
            lin_fim = st.number_input(f'Ponto final {i+1}:', min_value=lin_in, value=len(tabela))
            tipo_regressao = st.selectbox(f'Tipo de regressão {i+1}:', ['linear', 'log'], index=0)
            pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
        
        if st.button('Calcular Regressões'):
            calcular_regressao(tabela, num_regressoes, pontos_tipos)