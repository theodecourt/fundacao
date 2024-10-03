import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve

# Importa o currículo do arquivo curriculo.py
from curriculo import mostrar_curriculo

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

# Função principal para a página
def pagina_principal():
    st.title('Luciano Decourt')
    st.write("Bem-vindo ao site de Luciano Decourt")

# Função para a página "O engenheiro"
def pagina_engenheiro():
    st.title('Luciano Decourt')
    st.markdown(mostrar_curriculo(), unsafe_allow_html=True)

# Função para a página "Obras"
def pagina_obras():
    st.title('Obras de Luciano Decourt')
    st.write("Conteúdo de obras será adicionado aqui.")

# Função para a página "Artigos"
def pagina_artigos():
    st.title('Artigos de Luciano Decourt')
    st.write("Conteúdo de artigos será adicionado aqui.")

# Função para a página "Programas"
def pagina_programas():
    st.title('Programas de Luciano Decourt')
    st.write("Conteúdo de programas será adicionado aqui.")

# Função principal para a navegação
def main():
    st.sidebar.title("Navegação")
    pagina = st.sidebar.selectbox(
        "Selecione a página:",
        ["Página Principal", "O engenheiro", "Obras", "Artigos", "Programas"]
    )

    if pagina == "Página Principal":
        pagina_principal()
    elif pagina == "O engenheiro":
        pagina_engenheiro()
    elif pagina == "Obras":
        pagina_obras()
    elif pagina == "Artigos":
        pagina_artigos()
    elif pagina == "Programas":
        pagina_programas()

if __name__ == '__main__':
    main()
