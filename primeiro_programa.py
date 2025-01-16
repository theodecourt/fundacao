import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq
import streamlit as st
import plotly.express as px
import io

# Dicionário para números romanos
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

    label = "Baixar exemplo" if idioma == "Português" else "Download example"
    file_name = "exemplo.xlsx" if idioma == "Português" else "example.xlsx"
    st.download_button(
        label=label, 
        data=output, 
        file_name=file_name, 
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def carregar_tabela(idioma):
    uploaded_file = st.file_uploader(
        "Escolha o arquivo CSV ou XLSX" if idioma == "Português" else "Choose the CSV or XLSX file", 
        type=["csv", "xlsx"]
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, delimiter=';')
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None
    botao_download_exemplo(idioma)
    return None

def calcular_interseccao(reg1, reg2, tipo1, tipo2, x_min, x_max):
    """Tenta calcular a interseção entre duas regressões, se existir."""
    interseccoes = []
    
    if tipo1 == 'linear' and tipo2 == 'linear':
        # Regressões lineares: y = a1*x + b1 e y = a2*x + b2
        A = np.array([[reg1[0], -1], [reg2[0], -1]])
        B = np.array([-reg1[1], -reg2[1]])
        try:
            interseccao = np.linalg.solve(A, B)
            interseccoes.append(interseccao)
            st.write(f"Interseção Linear-Linear encontrada em x={interseccao[0]:.4f}, y={interseccao[1]:.4f}")
        except np.linalg.LinAlgError:
            st.write("Regressões lineares são paralelas ou não têm solução única.")
            return None  # Regressões paralelas ou sem solução
    
    elif tipo1 == 'log' and tipo2 == 'log':
        # Regressões logarítmicas (na forma: log(y) = a*log(x) + b)
        if reg1[0] == reg2[0]:
            st.write("Regressões logarítmicas são paralelas, sem interseção única.")
            return None
        log_x = (reg2[1] - reg1[1]) / (reg1[0] - reg2[0])
        x = 10 ** log_x
        y = 10 ** (reg1[0] * log_x + reg1[1])
        interseccao = [x, y]
        interseccoes.append(interseccao)
        st.write(f"Interseção Log-Log encontrada em x={x:.4f}, y={y:.4f}")
    
    elif (tipo1 == 'linear' and tipo2 == 'log') or (tipo1 == 'log' and tipo2 == 'linear'):
        # Regressão mista: uma linear e uma logarítmica
        if tipo1 == 'linear':
            reg_linear, reg_log = reg1, reg2
        else:
            reg_linear, reg_log = reg2, reg1
        
        def func_intersec(x):
            return reg_linear[0] * x + reg_linear[1] - 10**(reg_log[0] * np.log10(x) + reg_log[1])
        
        num_steps = 1000
        x_values = np.linspace(x_min, x_max, num_steps)
        f_values = func_intersec(x_values)
        
        # Identificar onde a função muda de sinal
        for i in range(len(x_values)-1):
            if np.isnan(f_values[i]) or np.isnan(f_values[i+1]):
                continue
            if f_values[i] * f_values[i+1] < 0:
                try:
                    raiz = brentq(func_intersec, x_values[i], x_values[i+1], xtol=1e-8)
                    if not any(np.isclose(raiz, r[0], atol=1e-6) for r in [e for e in interseccoes if e is not None]):
                        if raiz > 0:  # Garantir que a carga é positiva
                            y = reg_linear[0] * raiz + reg_linear[1]
                            interseccoes.append([raiz, y])
                            st.write(f"Interseção Linear-Log encontrada em x={raiz:.4f}, y={y:.4f}")
                except ValueError:
                    continue
    
    # Selecionar a interseção com o menor y (menor recalque)
    if interseccoes:
        interseccoes_validas = [inter for inter in interseccoes if inter is not None and inter[1] is not None]
        if not interseccoes_validas:
            st.write("Nenhuma interseção válida encontrada.")
            return None
        interseccao_selecionada = min(interseccoes_validas, key=lambda x: x[1])
        st.write(f"Interseção selecionada: x={interseccao_selecionada[0]:.4f}, y={interseccao_selecionada[1]:.4f}")
        return interseccao_selecionada
    else:
        st.write("Nenhuma interseção encontrada.")
        return None

def calcular_quc(reg, tipo_regressao, valor_critico):
    """Cálculo de Quc (carga para um determinado recalque) dependendo do tipo de regressão."""
    if tipo_regressao == 'linear':
        a = reg[1]
        b = reg[0]
        try:
            quc = a / ((1 / valor_critico) - b)
        except ZeroDivisionError:
            quc = np.nan
    else:  # log
        def func_quc_log(x):
            return 10**(reg[0] * np.log10(x) + reg[1]) - (x / valor_critico)
        try:
            quc = brentq(func_quc_log, 1e-2, 1e5)
        except ValueError:
            quc = np.nan
    return quc

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input, modo_plotagem):
    """
    :param modo_plotagem: String "Entre os pontos" ou "Até interseção"
    """
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)
    x0 = tabela['Carga']
    y0 = tabela['rigidez']

    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Numerar os pontos
    for i, (x, y) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    regressions = []
    tipos = []
    interseccoes = []

    # Valor de recalque crítico
    recalque_critico = 0.1 * diametro_estaca

    # Calcula cada regressão
    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        
        # Fatiando a tabela
        linear = tabela.iloc[lin_in: lin_fim + 1]
        
        if tipo_regressao == 'linear':
            reg = np.polyfit(linear['Carga'], linear['rigidez'], deg=1)
        else:  # log
            reg = np.polyfit(linear['logQ'], linear['logRig'], deg=1)
        
        regressions.append(reg)
        tipos.append(tipo_regressao)

        # Se já houver uma regressão anterior, tenta calcular a interseção
        if i > 0:
            x_min = max(tabela['Carga'].min(), tabela['Carga'].iloc[lin_in])
            x_max = tabela['Carga'].max()
            interseccao = calcular_interseccao(regressions[i-1], reg, tipos[i-1], tipo_regressao, x_min, x_max)
            interseccoes.append(interseccao)
        else:
            interseccoes.append(None)

    # Plotando cada regressão
    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        cor_texto = colors[i]

        st.markdown(
            f"<b style='color:{cor_texto};'>Pontos utilizados na regressão {num_romanos[i+1]}: {lin_in} até {lin_fim}</b>",
            unsafe_allow_html=True
        )

        # Definir x_inicio e x_fim dependendo do modo escolhido
        if modo_plotagem == "Entre os pontos":
            # Modo "reta entre os pontos"
            x_inicio = tabela['Carga'].iloc[lin_in]
            x_fim = tabela['Carga'].iloc[lin_fim]
        else:  # "Até interseção"
            if i == 0:
                x_inicio = tabela['Carga'].iloc[lin_in]
            else:
                interseccao_anterior = interseccoes[i]
                if interseccao_anterior is not None:
                    x_inicio = interseccao_anterior[0]
                else:
                    st.warning(
                        f"Não foi possível calcular a interseção entre as regressões {num_romanos[i]} e {num_romanos[i+1]}. "
                        "Usando ponto inicial da segmentação."
                    )
                    x_inicio = tabela['Carga'].iloc[lin_in]

            if i == num_regressoes - 1:
                x_fim = tabela['Carga'].iloc[lin_fim]
            else:
                interseccao_atual = interseccoes[i+1]
                if interseccao_atual is not None:
                    x_fim = interseccao_atual[0]
                else:
                    st.warning(
                        f"Não foi possível calcular a interseção entre as regressões {num_romanos[i+1]} e {num_romanos[i+2]}. "
                        "Usando ponto final da segmentação."
                    )
                    x_fim = tabela['Carga'].iloc[lin_fim]

        # Gera pontos de x
        x = np.linspace(x_inicio, x_fim, 100)
        
        # Calcular y
        if tipo_regressao == 'linear':
            predict = np.poly1d(regressions[i])
            y = predict(x)
            # Correlação linear
            corr_matrix = np.corrcoef(linear['rigidez'], linear['Carga'])
            equacao = f'rigidez (tf/mm) = {regressions[i][0]:.4f} * Carga (tf) + {regressions[i][1]:.4f}'
        else:
            predict = np.poly1d(regressions[i])
            y = 10**predict(np.log10(x))
            # Correlação para log
            corr_matrix = np.corrcoef(linear['logRig'], linear['logQ'])
            equacao = f'log(rigidez) = {regressions[i][0]:.4f} * log(Carga) + {regressions[i][1]:.4f}'

        corr = corr_matrix[0, 1]
        R_sq = corr**2

        # Cálculo de Quc
        quc = calcular_quc(regressions[i], tipo_regressao, recalque_critico)

        # Plot no Matplotlib
        plt.plot(x, y, color=colors[i], label=f'Regressão {i+1}' if idioma == 'Português' else f'Regression {i+1}')

        # Adicionar rótulo (número romano) no meio do trecho
        x_centro = (x_inicio + x_fim) / 2
        if tipo_regressao == 'linear':
            y_centro = predict(x_centro)
        else:
            y_centro = 10**predict(np.log10(x_centro))
        plt.text(
            x_centro, 
            y_centro * 1.15,
            f'{num_romanos[i+1]}', 
            color=colors[i], 
            fontsize=20, 
            fontweight='bold',
            ha='center'
        )
        
        # Exibir detalhes no Streamlit
        if idioma == "Português":
            st.write('Tipo de regressão:', tipo_regressao.capitalize())
            st.markdown(f'<span style="color:{cor_texto};"><strong>Equação da regressão:</strong> {equacao}</span>', unsafe_allow_html=True)
            st.write('R²:', R_sq)
            st.write(f'Quc para a regressão {num_romanos[i+1]}: {quc:.2f} tf')
        else:
            st.write('Regression type:', tipo_regressao.capitalize())
            st.markdown(f'<span style="color:{cor_texto};"><strong>Regression equation:</strong> {equacao}</span>', unsafe_allow_html=True)
            st.write('R²:', R_sq)
            st.write(f'Quc for regression {num_romanos[i+1]}: {quc:.2f} tf')

        # Se o usuário forneceu um recalque_input > 0, calcula a carga correspondente
        if recalque_input > 0:
            carga_calculada = calcular_quc(regressions[i], tipo_regressao, recalque_input)
            st.write(f"A carga para o recalque {recalque_input:.2f} mm é {carga_calculada:.2f} tf.")

        # Se o usuário forneceu um carga_input > 0, calcula o recalque correspondente
        if carga_input > 0:
            if tipo_regressao == 'linear':
                rigidez = predict(carga_input)
            else:
                rigidez = 10**predict(np.log10(carga_input))
            recalque_calculado = carga_input / rigidez
            st.write(f"Para a carga de {carga_input:.2f} tf, o recalque será {recalque_calculado:.2f} mm.")

    # Plotar interseções
    for idx, interseccao in enumerate(interseccoes):
        if interseccao is not None:
            st.markdown(
                f"<span style='color:black;'>Interseção entre regressão {num_romanos[idx]} e {num_romanos[idx+1]}: "
                f"Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}</span>",
                unsafe_allow_html=True
            )
            plt.plot(interseccao[0], interseccao[1], 'rx')  # 'x' vermelho

    # Finalizar o plot
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
        # Renomear colunas
        if "Carga (tf)" in tabela.columns and "Recalque (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Carga (tf)": "Carga", "Recalque (mm)": "Recalque"})
        elif "Load (tf)" in tabela.columns and "Settlement (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Load (tf)": "Carga", "Settlement (mm)": "Recalque"})
        else:
            st.error("Formato de coluna inválido. Certifique-se de que o arquivo contém 'Carga (tf)' e 'Recalque (mm)' ou 'Load (tf)' e 'Settlement (mm)'.")
            return
        
        # Forçar numérico
        try:
            tabela['Carga'] = tabela['Carga'].astype(float)
            tabela['Recalque'] = tabela['Recalque'].astype(float)
        except ValueError:
            st.error("As colunas 'Carga' e 'Recalque' devem conter apenas valores numéricos.")
            return

        # Calcular rigidez e log
        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ'] = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq'] = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig'] = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)', 
            min_value=0.01, format="%.2f"
        )

        # Inputs adicionais para cálculo
        recalque_input = st.number_input(
            'Quer calcular a carga para qual recalque? (mm)' if idioma == "Português" else 'Which settlement do you want to calculate load for? (mm)',
            format="%.2f", min_value=0.0
        )
        carga_input = st.number_input(
            'Quer estimar o recalque para qual carga? (tf)' if idioma == "Português" else 'Which load do you want to estimate settlement for? (tf)',
            format="%.2f", min_value=0.0
        )

        tabela = tabela.sort_values(by="Carga").reset_index(drop=True)

        # Gráfico 1: Carga vs Recalque
        fig = px.scatter(
            tabela, x="Carga", y="Recalque",
            labels={
                "Carga": "Carga (tf)", 
                "Recalque": "Recalque (mm)"
            } if idioma == "Português" else {
                "Carga": "Load (tf)", 
                "Recalque": "Settlement (mm)"
            }
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Carga vs Recalque" if idioma == "Português" else "Load vs Settlement",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Recalque (mm)" if idioma == "Português" else "Settlement (mm)"
        )
        for i, row in tabela.iterrows():
            fig.add_annotation(
                x=row["Carga"],
                y=row["Recalque"],
                text=str(i),
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-20
            )
        config_plotly = {
            "staticPlot": True,
            "displayModeBar": False
        }
        st.plotly_chart(fig, config=config_plotly)

        # Gráfico 2: Carga vs Rigidez
        fig2 = px.scatter(
            tabela, x="Carga", y="rigidez",
            labels={
                "Carga": "Carga (tf)", 
                "rigidez": "Rigidez (tf/mm)"
            } if idioma == "Português" else {
                "Carga": "Load (tf)", 
                "rigidez": "Stiffness (tf/mm)"
            }
        )
        fig2.update_layout(
            title="Carga vs Rigidez" if idioma == "Português" else "Load vs Stiffness",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Rigidez (tf/mm)" if idioma == "Português" else "Stiffness (tf/mm)"
        )
        for i, row in tabela.iterrows():
            fig2.add_annotation(
                x=row["Carga"],
                y=row["rigidez"],
                text=str(i),
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-20
            )
        st.plotly_chart(fig2, config=config_plotly)

        # Quantidade de regressões
        num_regressoes = st.selectbox(
            'Quantas regressões:' if idioma == "Português" else 'How many regressions?', 
            [1, 2, 3], index=0
        )

        # =======================
        # MODO DE PLOTAGEM
        # =======================
        modo_plotagem = st.selectbox(
            'Como deseja plotar as regressões?' if idioma == "Português" else 'How do you want to plot the regressions?',
            ['Entre os pontos', 'Até interseção'],
            index=0
        )
        # =======================

        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in_key = f'lin_in_{i}'
            lin_fim_key = f'lin_fim_{i}'
            tipo_regressao_key = f'tipo_regressao_{i}'

            lin_in_default = st.session_state.get(lin_in_key, '0')
            lin_fim_default = st.session_state.get(lin_fim_key, str(len(tabela) - 1))
            tipo_regressao_default = st.session_state.get(tipo_regressao_key, 'linear')

            lin_in_str = st.text_input(
                f'Ponto inicial da regressão {num_romanos[i+1]} (0-based):' if idioma == "Português" else f'Starting point of regression {num_romanos[i+1]} (0-based):', 
                value=lin_in_default,
                key=lin_in_key
            )

            lin_fim_str = st.text_input(
                f'Ponto final da regressão {num_romanos[i+1]} (0-based):' if idioma == "Português" else f'Ending point of regression {num_romanos[i+1]} (0-based):', 
                value=lin_fim_default,
                key=lin_fim_key
            )

            try:
                lin_in = int(lin_in_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto inicial da regressão {num_romanos[i+1]}. Insira um número inteiro.")
                return
            try:
                lin_fim = int(lin_fim_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto final da regressão {num_romanos[i+1]}. Insira um número inteiro.")
                return

            if lin_in < 0 or lin_in >= len(tabela):
                st.error(f"Ponto inicial da regressão {num_romanos[i+1]} deve estar entre 0 e {len(tabela)-1}.")
                return
            if lin_fim < lin_in or lin_fim >= len(tabela):
                st.error(f"Ponto final da regressão {num_romanos[i+1]} deve estar entre {lin_in} e {len(tabela)-1}.")
                return

            tipo_regressao = st.selectbox(
                f'Tipo de regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Regression type {num_romanos[i+1]}:', 
                ['linear', 'log'], 
                index=0,
                key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
        
        # Botão para calcular
        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao(
                tabela, num_regressoes, pontos_tipos, 
                diametro_estaca, idioma, 
                carga_input, recalque_input,
                modo_plotagem  # <-- Passamos a escolha do modo para a função
            )

# Rode seu programa definindo o idioma desejado
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)
