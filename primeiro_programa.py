import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import streamlit as st
import plotly.express as px
import io

num_romanos = {1: 'I', 2: 'II', 3: 'III'}

def criar_tabela_exemplo(idioma):
    dados = {
        "Carga (tf)": [1200, 1125, 1050, 975, 900, 825, 750, 675, 600, 525, 450, 375, 300, 225, 150, 75],
        "Recalque (mm)": [27.21, 24.55, 21.95, 19.35, 17.28, 14.72, 12.81, 11.03, 9.52, 8.30, 6.92, 5.19, 3.79, 2.48, 1.51, 0.66]
    } if idioma == "Português" else {
        "Load (tf)": [1200, 1125, 1050, 975, 900, 825, 750, 675, 600, 525, 450, 375, 300, 225, 150, 75],
        "Settlement (mm)": [27.21, 24.55, 21.95, 19.35, 17.28, 14.72, 12.81, 11.03, 9.52, 8.30, 6.92, 5.19, 3.79, 2.48, 1.51, 0.66]
    }
    return pd.DataFrame(dados)

def botao_download_exemplo(idioma):
    tabela_exemplo = criar_tabela_exemplo(idioma)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tabela_exemplo.to_excel(writer, index=False, sheet_name='Exemplo')
    output.seek(0)
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

    label = "Baixando exemplo" if idioma == "Português" else "Downloading example"
    file_name = "exemplo.xlsx" if idioma == "Português" else "example.xlsx"
    st.download_button(label=label, data=output, file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def carregar_tabela(idioma):
    uploaded_file = st.file_uploader(
        "Escolha o arquivo CSV ou XLSX" if idioma == "Português" else "Choose the CSV or XLSX file", 
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, delimiter=';')
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    botao_download_exemplo(idioma)
    return None

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

def calcular_quc(reg, tipo_regressao, valor_critico):
    if tipo_regressao == 'linear':
        a = reg[1]
        b = reg[0]
        quc = a / ((1 / valor_critico) - b)
    else:  # log
        def func_quc_log(x):
            return 10**(reg[0] * np.log10(x) + reg[1]) - (x / valor_critico)
        quc = fsolve(func_quc_log, x0=1)[0]
    return quc

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    # Sort the data by 'Carga' in ascending order
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)
    x0 = tabela['Carga']
    y0 = tabela['rigidez']

    colors = ['blue', 'red', 'green']  # Use color names for consistency
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Number the data points
    for i, (x, y) in enumerate(zip(x0, y0), start=1):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    regressions = []
    tipos = []
    interseccoes = []

    recalque_critico = 0.1 * diametro_estaca

    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        linear = tabela[lin_in-1:lin_fim]
        
        if tipo_regressao == 'linear':
            reg = np.polyfit(linear['Carga'], linear['rigidez'], deg=1)
        else:  # log
            reg = np.polyfit(linear['logQ'], linear['logRig'], deg=1)

        regressions.append(reg)
        tipos.append(tipo_regressao)

        if i > 0:
            interseccao = calcular_interseccao(regressions[i-1], reg, tipos[i-1], tipo_regressao)
            interseccoes.append(interseccao)

    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        linear = tabela[lin_in-1:lin_fim]

        # Define the color based on regression
        cor_texto = colors[i]  # Use the color from the colors list
        
        st.markdown(
            f"<b style='color:{cor_texto};'>Pontos utilizados na regressão {num_romanos[i+1]}: {lin_in} até {lin_fim}</b>",
            unsafe_allow_html=True
        )

        if tipo_regressao == 'linear':
            x_inicio = tabela['Carga'].iloc[lin_in-1] if i == 0 else interseccoes[i-1][0]
            x_fim = tabela['Carga'].iloc[lin_fim-1] if i == num_regressoes-1 else interseccoes[i][0]
            x = np.linspace(x_inicio, x_fim, 100)
            predict = np.poly1d(regressions[i])
            y = predict(x)
            corr_matrix = np.corrcoef(linear['rigidez'], linear['Carga'])
            equacao = f'rigidez (tf/mm) = {regressions[i][0]:.4f} * Carga (tf) + {regressions[i][1]:.4f}'

        else:  # log
            x_inicio = tabela['Carga'].iloc[lin_in-1] if i == 0 else interseccoes[i-1][0]
            x_fim = tabela['Carga'].iloc[lin_fim-1] if i == num_regressoes-1 else interseccoes[i][0]
            x = np.linspace(x_inicio, x_fim, 100)
            predict = np.poly1d(regressions[i])
            y = 10**predict(np.log10(x))
            corr_matrix = np.corrcoef(linear['logRig'], linear['logQ'])
            equacao = f'log(rigidez) = {regressions[i][0]:.4f} * log(Carga) + {regressions[i][1]:.4f}'

        corr = corr_matrix[0, 1]
        R_sq = corr**2

        quc = calcular_quc(regressions[i], tipo_regressao, recalque_critico)

        plt.plot(x, y, color=colors[i], label=f'Regressão {i+1}' if idioma == 'Português' else f'Regression {i+1}')
        
        if idioma == "Português":
            st.write('Tipo de regressão:', tipo_regressao.capitalize())
            # Display the regression equation in the same color as the regression line
            st.markdown(f'<span style="color:{cor_texto};"><strong>Equação da regressão:</strong> {equacao}</span>', unsafe_allow_html=True)  # Modified line
            st.write('R²:', R_sq)
            st.write(f'Quc para a regressão {num_romanos[i+1]}: {quc:.2f} tf')

        else:
            st.write('Regression type:', tipo_regressao.capitalize())
            # Display the regression equation in the same color as the regression line
            st.markdown(f'<span style="color:{cor_texto};"><strong>Regression equation:</strong> {equacao}</span>', unsafe_allow_html=True)  # Modified line
            st.write('R²:', R_sq)
            st.write(f'Quc for regression {num_romanos[i+1]}: {quc:.2f} tf')

        # Calculate and display load and settlement based on regression
        if recalque_input > 0:
            carga_calculada = calcular_quc(regressions[i], tipo_regressao, recalque_input)
            st.write(f"A carga para o recalque {recalque_input:.2f} mm é {carga_calculada:.2f} tf.")
        
        if carga_input > 0:
            # Calculate stiffness for the given load
            if tipo_regressao == 'linear':
                rigidez = predict(carga_input)
            else:
                rigidez = 10**predict(np.log10(carga_input))
            recalque_calculado = carga_input / rigidez
            st.write(f"Para a carga de {carga_input:.2f} tf, o recalque será {recalque_calculado:.2f} mm.")

    if interseccoes:
        for idx, interseccao in enumerate(interseccoes):
            st.markdown(
                f"<span style='color:black;'>Interseção entre regressão {num_romanos[idx+1]} e regressão {num_romanos[idx+2]}: Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}</span>",
                unsafe_allow_html=True
            )
            plt.plot(interseccao[0], interseccao[1], 'rx')  # Mark the intersection with a red 'x'

    if idioma == "Português":
        plt.xlabel('Carga (tf)')
        plt.ylabel('Rigidez (tf/mm)')
        plt.title('Regressão de Carga x Rigidez')
    else:
        plt.xlabel('Load (tf)')
        plt.ylabel('Stiffness (tf/mm)')
        plt.title('Load vs Stiffness Regression')

    plt.legend(loc='best')
    st.pyplot(plt)
    
def primeiro_programa(idioma):
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        if "Carga (tf)" in tabela.columns and "Recalque (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Carga (tf)": "Carga", "Recalque (mm)": "Recalque"})
        else:
            if "Load (tf)" in tabela.columns and "Settlement (mm)" in tabela.columns:
                tabela = tabela.rename(columns={"Load (tf)": "Carga", "Settlement (mm)": "Recalque"})
        
        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)', 
            min_value=0.01, format="%.2f"
        )

        # Additional inputs for load and settlement
        recalque_input = st.number_input('Quer calcular a carga para qual recalque? (mm)', format="%.2f", min_value=0.0)
        carga_input = st.number_input('Quer estimar o recalque para qual carga? (tf)', format="%.2f", min_value=0.0)

        # Primeiro gráfico
        fig = px.scatter(
            tabela, x="Carga", y="Recalque",
            labels={"Carga": "Carga (tf)", "Recalque": "Recalque (mm)"} if idioma == "Português" else {"Carga": "Load (tf)", "Recalque": "Settlement (mm)"}
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Carga vs Recalque" if idioma == "Português" else "Load vs Settlement",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Recalque (mm)" if idioma == "Português" else "Settlement (mm)"
        )

        # Adicionar numeração dos pontos
        for i, row in tabela.iterrows():
            fig.add_annotation(
                x=row["Carga"],
                y=row["Recalque"],
                text=str(i + 1),
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-20
            )

        st.plotly_chart(fig)

        tabela['rigidez'] = tabela.apply(lambda row: row.Carga / row.Recalque, axis=1)
        fig2 = px.scatter(tabela, x="Carga", y="rigidez")
        fig2.update_layout(
            title="Carga vs Rigidez" if idioma == "Português" else "Load vs Stiffness",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Rigidez (tf/mm)" if idioma == "Português" else "Stiffness (tf/mm)"
        )
        st.plotly_chart(fig2)

        tabela['logQ'] = tabela.apply(lambda row: math.log(row.Carga, 10), axis=1)
        tabela['logReq'] = tabela.apply(lambda row: math.log(row.Recalque, 10), axis=1)
        tabela['logRig'] = tabela.apply(lambda row: math.log(row.rigidez, 10), axis=1)

        num_regressoes = st.selectbox(
            'Quantas regressões:' if idioma == "Português" else 'How many regressions?', 
            [1, 2, 3], index=0
        )

        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in_key = f'lin_in_{i}'
            lin_fim_key = f'lin_fim_{i}'
            tipo_regressao_key = f'tipo_regressao_{i}'

            # Retrieve previous values from session state or set defaults
            lin_in_default = st.session_state.get(lin_in_key, '1')
            lin_fim_default = st.session_state.get(lin_fim_key, str(len(tabela)))
            tipo_regressao_default = st.session_state.get(tipo_regressao_key, 'linear')

            lin_in_str = st.text_input(
                f'Ponto inicial da regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Starting point of regression {num_romanos[i+1]}:', 
                value=lin_in_default,
                key=lin_in_key
            )

            lin_fim_str = st.text_input(
                f'Ponto final da regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Ending point of regression {num_romanos[i+1]}:', 
                value=lin_fim_default,
                key=lin_fim_key
            )

            # Parse the input strings to integers, handle errors
            try:
                lin_in = int(lin_in_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto inicial da regressão {num_romanos[i+1]}. Por favor, insira um número inteiro.")
                return

            try:
                lin_fim = int(lin_fim_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto final da regressão {num_romanos[i+1]}. Por favor, insira um número inteiro.")
                return

            # Check if lin_in and lin_fim are within valid range
            if lin_in < 1 or lin_in > len(tabela):
                st.error(f"Ponto inicial da regressão {num_romanos[i+1]} deve estar entre 1 e {len(tabela)}.")
                return
            if lin_fim < lin_in or lin_fim > len(tabela):
                st.error(f"Ponto final da regressão {num_romanos[i+1]} deve estar entre {lin_in} e {len(tabela)}.")
                return

            tipo_regressao = st.selectbox(
                f'Tipo de regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Regression type {num_romanos[i+1]}:', 
                ['linear', 'log'], index=0,
                key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
        
        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input)

idioma = 'Português'  # or 'English'
primeiro_programa(idioma)