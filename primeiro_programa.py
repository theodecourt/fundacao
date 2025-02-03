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
    """Tenta calcular a interseção entre duas regressões, se existir, dentro do intervalo [x_min, x_max]."""
    interseccoes = []
    
    # ---- FUNÇÕES AUXILIARES PARA CHECAR SE x ESTÁ NO DOMÍNIO E SE É POSITIVO ----
    def esta_no_intervalo(x):
        return (x_min <= x <= x_max)
    
    def esta_no_intervalo_positivo(x):
        # Se quisermos garantir x > 0, e dentro do [x_min, x_max]
        return (x > 0) and (x_min <= x <= x_max)
    # ---------------------------------------------------------------------------

    # =============== CASO 1: Ambas lineares ===============
    if tipo1 == 'linear' and tipo2 == 'linear':
        # Regressões lineares: y = a1*x + b1 e y = a2*x + b2
        # reg = [a, b], onde y = a*x + b
        a1, b1 = reg1[0], reg1[1]
        a2, b2 = reg2[0], reg2[1]

        # Se a1 == a2, são paralelas (ou coincidentes) => sem interseção única
        if np.isclose(a1, a2, atol=1e-12):
            st.write("Regressões lineares são paralelas ou coincidentes, sem interseção única.")
            return None

        # (a1 - a2)*x = (b2 - b1) => x = (b2 - b1)/(a1 - a2)
        x_int = (b2 - b1) / (a1 - a2)
        y_int = a1*x_int + b1

        # Checar se está dentro do intervalo
        if esta_no_intervalo(x_int):
            interseccoes.append([x_int, y_int])
            st.write(f"Interseção Linear-Linear em x={x_int:.4f}, y={y_int:.4f}")
        else:
            st.write("Interseção Linear-Linear está fora do intervalo definido.")
    
    # =============== CASO 2: Ambas logarítmicas (log-log) ===============
    elif tipo1 == 'log' and tipo2 == 'log':
        # No código, reg = [a, b] => log(rigidez) = a*log(carga) + b
        # rigidez = 10^(a*log10(carga) + b)
        a1, b1 = reg1
        a2, b2 = reg2

        # a1*logQ + b1 = a2*logQ + b2 => (a1 - a2)*logQ = b2 - b1 => logQ = (b2 - b1)/(a1 - a2)
        # Q = 10^(logQ)
        if np.isclose(a1, a2, atol=1e-12):
            st.write("Regressões logarítmicas são paralelas, sem interseção única.")
            return None
        
        log_x = (b2 - b1) / (a1 - a2)
        x_int = 10**(log_x)
        # y => rigidez = 10^(a1 * log_x + b1)
        y_int = 10**(a1 * log_x + b1)

        if esta_no_intervalo_positivo(x_int):
            interseccoes.append([x_int, y_int])
            st.write(f"Interseção Log-Log em x={x_int:.4f}, y={y_int:.4f}")
        else:
            st.write("Interseção Log-Log está fora do intervalo ou x ≤ 0.")
    
    # =============== CASO 3: Linear + Log ===============
    elif (tipo1 == 'linear' and tipo2 == 'log') or (tipo1 == 'log' and tipo2 == 'linear'):
        # Precisamos montar a função de diferença e usar método numérico
        if tipo1 == 'linear':
            a_lin, b_lin = reg1  # y_lin = a_lin*x + b_lin
            a_log, b_log = reg2  # y_log = 10^(a_log*log10(x) + b_log)
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1
        
        def func_intersec(x):
            # linear - log
            val_lin = a_lin*x + b_lin
            val_log = 10**(a_log * np.log10(x) + b_log) if x > 0 else np.nan
            return val_lin - val_log
        
        # ### CORREÇÃO PRINCIPAL ###
        # Vamos discretizar [x_min, x_max], procurar mudanças de sinal e usar brentq localmente.
        num_steps = 200
        xs = np.linspace(x_min, x_max, num_steps)
        fs = []
        for xi in xs:
            if xi <= 0:
                fs.append(np.nan)
            else:
                fs.append(func_intersec(xi))
        
        # Identificar onde a função muda de sinal
        for i in range(len(xs)-1):
            f1, f2 = fs[i], fs[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 * f2 < 0:  # mudança de sinal
                try:
                    raiz = brentq(func_intersec, xs[i], xs[i+1], xtol=1e-8)
                    if esta_no_intervalo_positivo(raiz):
                        y_raiz = a_lin*raiz + b_lin
                        interseccoes.append([raiz, y_raiz])
                        st.write(f"Interseção Linear-Log em x={raiz:.4f}, y={y_raiz:.4f}")
                except ValueError:
                    continue
    
    # =============== Selecionar interseção final, se houver ===============
    if interseccoes:
        # Escolher a com menor y (por convenção do código) ou simplesmente a primeira
        interseccao_selecionada = min(interseccoes, key=lambda x: x[1])
        st.write(f"Interseção selecionada: x={interseccao_selecionada[0]:.4f}, y={interseccao_selecionada[1]:.4f}")
        return interseccao_selecionada
    else:
        st.write("Nenhuma interseção encontrada (ou fora do intervalo).")
        return None

def calcular_quc(reg, tipo_regressao, valor_critico):
    """Cálculo de Quc (carga para um determinado recalque) dependendo do tipo de regressão."""
    # Nota: Esta função continua como estava no seu código original
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
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)
    x0 = tabela['Carga']
    y0 = tabela['rigidez']

    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    regressions = []
    tipos = []
    interseccoes = []

    recalque_critico = 0.1 * diametro_estaca

    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        subset = tabela.iloc[lin_in: lin_fim + 1]

        if tipo_regressao == 'linear':
            reg = np.polyfit(subset['Carga'], subset['rigidez'], deg=1)
            equacao = f'rigidez = {reg[0]:.4f} * Carga + {reg[1]:.4f}'
        else:
            reg = np.polyfit(subset['logQ'], subset['logRig'], deg=1)
            equacao = f'log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}'

        regressions.append(reg)
        tipos.append(tipo_regressao)

        if i > 0:
            prev_lin_in = pontos_tipos[i-1][0]
            prev_lin_fim = pontos_tipos[i-1][1]
            subset_prev = tabela.iloc[prev_lin_in: prev_lin_fim + 1]

            x_min_global = max(subset['Carga'].min(), subset_prev['Carga'].min())
            x_max_global = min(subset['Carga'].max(), subset_prev['Carga'].max())

            if x_min_global < x_max_global:
                interseccao = calcular_interseccao(
                    regressions[i-1], reg, tipos[i-1], tipo_regressao, x_min_global, x_max_global
                )
                interseccoes.append(interseccao)
            else:
                interseccoes.append(None)
        else:
            interseccoes.append(None)

        intersecao_str = ''
        if interseccoes[i] is not None:
            intersecao_str = f' (Interseção: x={interseccoes[i][0]:.4f}, y={interseccoes[i][1]:.4f})'

        st.markdown(
            f'<span style="color:{colors[i]};"><strong>Equação da regressão {num_romanos[i+1]}:</strong> {equacao}{intersecao_str}</span>',
            unsafe_allow_html=True
        )


def primeiro_programa(idioma):
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        # Renomear colunas conforme língua
        if "Carga (tf)" in tabela.columns and "Recalque (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Carga (tf)": "Carga", "Recalque (mm)": "Recalque"})
        elif "Load (tf)" in tabela.columns and "Settlement (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Load (tf)": "Carga", "Settlement (mm)": "Recalque"})
        else:
            st.error("Formato de coluna inválido. Certifique-se de que o arquivo contém 'Carga (tf)' e 'Recalque (mm)' ou 'Load (tf)' e 'Settlement (mm)'.")
            return
        
        # Converter para numérico
        try:
            tabela['Carga'] = tabela['Carga'].astype(float)
            tabela['Recalque'] = tabela['Recalque'].astype(float)
        except ValueError:
            st.error("As colunas 'Carga' e 'Recalque' devem conter apenas valores numéricos.")
            return

        # Calcular rigidez e logs
        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ'] = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq'] = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig'] = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)', 
            min_value=0.01, format="%.2f"
        )

        # Inputs de cálculo
        recalque_input = st.number_input(
            'Quer calcular a carga para qual recalque? (mm)' if idioma == "Português" else 'Which settlement do you want to calculate load for? (mm)',
            format="%.2f", min_value=0.0
        )
        carga_input = st.number_input(
            'Quer estimar o recalque para qual carga? (tf)' if idioma == "Português" else 'Which load do you want to estimate settlement for? (tf)',
            format="%.2f", min_value=0.0
        )

        tabela = tabela.sort_values(by="Carga").reset_index(drop=True)

        # Gráfico 1: Carga vs Recalque (Plotly)
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

        # Gráfico 2: Carga vs Rigidez (Plotly)
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

        # Escolher quantas regressões
        num_regressoes = st.selectbox(
            'Quantas regressões:' if idioma == "Português" else 'How many regressions?', 
            [1, 2, 3], index=0
        )

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
                lin_in_val = int(lin_in_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto inicial da regressão {num_romanos[i+1]}. Insira um número inteiro.")
                return
            try:
                lin_fim_val = int(lin_fim_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto final da regressão {num_romanos[i+1]}. Insira um número inteiro.")
                return

            if lin_in_val < 0 or lin_in_val >= len(tabela):
                st.error(f"Ponto inicial da regressão {num_romanos[i+1]} deve estar entre 0 e {len(tabela)-1}.")
                return
            if lin_fim_val < lin_in_val or lin_fim_val >= len(tabela):
                st.error(f"Ponto final da regressão {num_romanos[i+1]} deve estar entre {lin_in_val} e {len(tabela)-1}.")
                return

            tipo_reg_val = st.selectbox(
                f'Tipo de regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Regression type {num_romanos[i+1]}:', 
                ['linear', 'log'], 
                index=0,
                key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        modo_plotagem = st.selectbox(
            'Como deseja plotar as regressões?' if idioma == "Português" else 'How do you want to plot the regressions?',
            ['Até interseção', 'Entre os pontos'],
            index=0
        )

        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao(
                tabela, num_regressoes, pontos_tipos, 
                diametro_estaca, idioma, 
                carga_input, recalque_input,
                modo_plotagem
            )

# Rode seu programa definindo o idioma desejado
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)