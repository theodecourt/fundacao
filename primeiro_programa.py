import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import streamlit as st
import io

# Função para criar o dataframe de exemplo
def criar_tabela_exemplo(idioma):
    if idioma == "Português":
        dados = {
            "Carga (tf)": [1200, 1125, 1050, 975, 900, 825, 750, 675, 600, 525, 450, 375, 300, 225, 150, 75],
            "Recalque (mm)": [27.21, 24.55, 21.95, 19.35, 17.28, 14.72, 12.81, 11.03, 9.52, 8.30, 6.92, 5.19, 3.79, 2.48, 1.51, 0.66]
        }
    else:
        dados = {
            "Load (tf)": [1200, 1125, 1050, 975, 900, 825, 750, 675, 600, 525, 450, 375, 300, 225, 150, 75],
            "Settlement (mm)": [27.21, 24.55, 21.95, 19.35, 17.28, 14.72, 12.81, 11.03, 9.52, 8.30, 6.92, 5.19, 3.79, 2.48, 1.51, 0.66]
        }
    return pd.DataFrame(dados)

# Função para gerar o botão de download de arquivo XLSX
def botao_download_exemplo(idioma):
    # Cria a tabela de exemplo
    tabela_exemplo = criar_tabela_exemplo(idioma)

    # Converte o dataframe para Excel
    output = io.BytesIO()  # Um buffer em memória para o arquivo Excel
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tabela_exemplo.to_excel(writer, index=False, sheet_name='Exemplo')

    # Move o ponteiro do buffer para o início do arquivo
    output.seek(0)

    # Adiciona estilo para destacar o botão de download
    st.markdown(
        """
        <style>
        .stDownloadButton button {
            background-color: #FFC300;
            color: black;
            font-weight: bold;
        }
        .stDownloadButton button:hover {
            background-color: #FFB000;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

    # Botão de download com estilo personalizado
    st.download_button(
        label="Baixando exemplo" if idioma == "Português" else "Downloading example",
        data=output,
        file_name="exemplo.xlsx" if idioma == "Português" else "example.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Função para carregar a tabela
def carregar_tabela(idioma):
    # Aceita arquivos CSV e XLSX
    uploaded_file = st.file_uploader("Escolha o arquivo CSV ou XLSX" if idioma == "Português" else "Choose the CSV or XLSX file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Verifica o tipo de arquivo e carrega o arquivo corretamente
        if uploaded_file.name.endswith('.csv'):
            tabela = pd.read_csv(uploaded_file, delimiter=';')
        elif uploaded_file.name.endswith('.xlsx'):
            tabela = pd.read_excel(uploaded_file)

        # Verifica o idioma e ajusta as colunas
        if idioma == "Português":
            if "Carga" in tabela.columns and "Recalque" in tabela.columns:
                tabela.columns = ["Carga", "Recalque"]
        else:
            if "Load" in tabela.columns and "Settlement" in tabela.columns:
                tabela.columns = ["Carga", "Recalque"]  # Renomeia para usar os mesmos nomes internamente

        return tabela

    st.title('Baixando exemplo' if idioma == "Português" else 'Downloading example')
    botao_download_exemplo(idioma)

# Função para calcular a intersecção entre duas regressões
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

# Função para calcular o Quc
def calcular_quc(reg, tipo_regressao, recalque_critico):
    if tipo_regressao == 'linear':
        a = reg[1]
        b = reg[0]
        quc = a / ((1 / recalque_critico) - b)
    else:  # log
        def func_quc_log(x):
            return 10**(reg[0] * np.log10(x) + reg[1]) - (x / recalque_critico)
        quc = fsolve(func_quc_log, x0=1)[0]
    return quc

# Função para calcular a regressão e plotar os gráficos
def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma):
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    
    colors = ['b', 'red', 'green']
    plt.plot(x0, y0, 'go', label='Dados Originais')
    
    regressions = []
    tipos = []
    
    recalque_critico = 0.1 * diametro_estaca  # Cálculo do recalque crítico

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
            x = np.linspace(0.1, tabela['Carga'].max(), 100)  # Evitar log(0)
            y = 10**predict(np.log10(x))
            corr_matrix = np.corrcoef(linear['logRig'], linear['logQ'])
            equacao = f'log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}'
        
        corr = corr_matrix[0, 1]
        R_sq = corr**2

        # Calcular o Quc
        quc = calcular_quc(reg, tipo_regressao, recalque_critico)

        plt.plot(x, y, colors[i], label=f'Regressão {i+1}')
        
        if idioma == "Português":
            st.write(f'Pontos utilizados na regressão {i+1}: {lin_in} até {lin_fim}')
            st.write('Tipo de regressão:', tipo_regressao.capitalize())
            st.write('Equação da regressão:', equacao)
            st.write('R²:', R_sq)
            st.write(f'Quc para a regressão {i+1}: {quc:.4f} tf')
        else:
            st.write(f'Points used in regression {i+1}: {lin_in} to {lin_fim}')
            st.write('Regression type:', tipo_regressao.capitalize())
            st.write('Regression equation:', equacao)
            st.write('R²:', R_sq)
            st.write(f'Quc for regression {i+1}: {quc:.4f} tf')


        # Adiciona a regressão e o tipo de regressão à lista
        regressions.append(reg)
        tipos.append(tipo_regressao)

    # Verificar se há pelo menos duas regressões para calcular a interseção
    if len(regressions) >= 2:
        # Calcular e mostrar pontos de interseção entre todas as combinações possíveis
        for i in range(num_regressoes):
            for j in range(i + 1, num_regressoes):
                if i < len(regressions) and j < len(regressions):
                    interseccao = calcular_interseccao(regressions[i], regressions[j], tipos[i], tipos[j])
                    plt.plot(interseccao[0], interseccao[1], 'rx')  # Marca a interseção com um 'x' vermelho
                    st.write(f'Interseção entre regressão {i+1} e {j+1}: Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}')
    
    if idioma == "Português":
        plt.xlabel('Carga')
        plt.ylabel('Rigidez')
        plt.title('Regressão de Carga x Rigidez')
    else:
        plt.xlabel('Load')
        plt.ylabel('Stiffness')
        plt.title('Load vs Stiffness Regression')

    plt.legend().set_visible(False)  # Oculta a caixa de legenda
    st.pyplot(plt)

# Função principal para executar o fluxo
def primeiro_programa(idioma):
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        # Pergunta o diâmetro da estaca
        diametro_estaca = st.number_input('Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)', min_value=0.01, format="%.2f")

        # Plota os gráficos antes de exibir as opções de regressões
        fig = px.scatter(tabela, x="Carga", y="Recalque")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Regressão de Carga x Recalque" if idioma == "Português" else "Load vs Settlement Regression",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Recalque (mm)" if idioma == "Português" else "Settlement (mm)"
        )
        st.plotly_chart(fig)

        # Calcula e exibe rigidez
        tabela['rigidez'] = tabela.apply(lambda row: row.Carga / row.Recalque, axis=1)
        fig2 = px.scatter(tabela, x="Carga", y="rigidez")
        fig2.update_layout(
            title="Regressão de Carga x Rigidez" if idioma == "Português" else "Load vs Stiffness Regression",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Rigidez (tf/mm)" if idioma == "Português" else "Stiffness (tf/mm)"
        )
        st.plotly_chart(fig2)

        tabela['logQ'] = tabela.apply(lambda row: math.log(row.Carga, 10), axis=1)
        tabela['logReq'] = tabela.apply(lambda row: math.log(row.Recalque, 10), axis=1)
        tabela['logRig'] = tabela.apply(lambda row: math.log(row.rigidez, 10), axis=1)
        
        # Seletor para o número de regressões
        num_regressoes = st.selectbox('Quantas regressões:' if idioma == "Português" else 'How many regressions?', [1, 2, 3], index=0)
        
        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in = st.number_input(f'Ponto inicial {i+1}:' if idioma == "Português" else f'Starting point {i+1}:', min_value=1, max_value=len(tabela), value=1)
            lin_fim = st.number_input(f'Ponto final {i+1}:' if idioma == "Português" else f'Ending point {i+1}:', min_value=lin_in, max_value=len(tabela), value=len(tabela))
            tipo_regressao = st.selectbox(f'Tipo de regressão {i+1}:' if idioma == "Português" else f'Regression type {i+1}:', ['linear', 'log'], index=0)
            pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
        
        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma)
