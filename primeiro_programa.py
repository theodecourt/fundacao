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
        """,
        unsafe_allow_html=True
    )

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
    """Tenta calcular a interseção entre duas regressões dentro do intervalo [x_min, x_max]."""
    interseccoes = []

    def esta_no_intervalo(x):
        return (x_min <= x <= x_max)
    
    def esta_no_intervalo_positivo(x):
        return (x > 0) and (x_min <= x <= x_max)
    
    # Caso 1: Ambas lineares
    if tipo1 == 'linear' and tipo2 == 'linear':
        a1, b1 = reg1[0], reg1[1]
        a2, b2 = reg2[0], reg2[1]
        if np.isclose(a1, a2, atol=1e-12):
            return None  # Paralelas ou coincidentes
        x_int = (b2 - b1) / (a1 - a2)
        y_int = a1*x_int + b1
        if esta_no_intervalo(x_int):
            interseccoes.append([x_int, y_int])

    # Caso 2: Ambas log
    elif tipo1 == 'log' and tipo2 == 'log':
        a1, b1 = reg1
        a2, b2 = reg2
        if np.isclose(a1, a2, atol=1e-12):
            return None
        log_x = (b2 - b1) / (a1 - a2)
        x_int = 10**(log_x)
        y_int = 10**(a1*log_x + b1)
        if esta_no_intervalo_positivo(x_int):
            interseccoes.append([x_int, y_int])

    # Caso 3: Linear + Log
    else:
        if tipo1 == 'linear':
            a_lin, b_lin = reg1
            a_log, b_log = reg2
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1

        def func_intersec(x):
            val_lin = a_lin*x + b_lin
            val_log = 10**(a_log * np.log10(x) + b_log) if x > 0 else np.nan
            return val_lin - val_log
        
        num_steps = 200
        xs = np.linspace(x_min, x_max, num_steps)
        fs = []
        for xi in xs:
            if xi <= 0:
                fs.append(np.nan)
            else:
                fs.append(func_intersec(xi))

        for i in range(len(xs)-1):
            f1, f2 = fs[i], fs[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 * f2 < 0:
                try:
                    raiz = brentq(func_intersec, xs[i], xs[i+1], xtol=1e-8)
                    if esta_no_intervalo_positivo(raiz):
                        # Valor de y na interseção
                        if tipo1 == 'linear':
                            y_raiz = a_lin*raiz + b_lin
                        else:
                            y_raiz = 10**(a_log*np.log10(raiz) + b_log)
                        interseccoes.append([raiz, y_raiz])
                except ValueError:
                    continue

    if interseccoes:
        # Caso existam várias raízes, retornamos a de menor x ou menor y?
        # Aqui, por simplicidade, retornamos a primeira (ou poderíamos retornar todas).
        # Vamos retornar a que tiver menor x, por exemplo:
        return min(interseccoes, key=lambda x: x[0])
    else:
        return None

def calcular_quc(reg, tipo_regressao, valor_critico):
    """Cálculo de Quc (carga para um determinado recalque)."""
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

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    Agora as regressões são sempre plotadas do mínimo ao máximo de carga, 
    e marcamos X em toda interseção encontrada entre quaisquer duas regressões.
    """
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)
    x0 = tabela['Carga']
    y0 = tabela['rigidez']

    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Numerar os pontos no gráfico
    for i, (x, y) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    # Armazena os resultados de cada regressão
    regressions = []
    tipos = []
    # Para sabermos o min e max de cada trecho
    intervals = []

    # Valor de recalque crítico
    recalque_critico = 0.1 * diametro_estaca

    # Intervalo de plotagem global
    x_min_global = tabela['Carga'].min()
    x_max_global = tabela['Carga'].max()

    # 1) CALCULAR REGRESSÕES
    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]
        
        subset = tabela.iloc[lin_in: lin_fim + 1]

        if tipo_regressao == 'linear':
            reg = np.polyfit(subset['Carga'], subset['rigidez'], deg=1)
            p = np.poly1d(reg)
            equacao = f'rigidez = {reg[0]:.4f} * Carga + {reg[1]:.4f}'
        else:
            reg = np.polyfit(subset['logQ'], subset['logRig'], deg=1)
            p = np.poly1d(reg)
            equacao = f'log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}'

        # Armazenar
        regressions.append(reg)
        tipos.append(tipo_regressao)
        intervals.append((subset['Carga'].min(), subset['Carga'].max()))

        # Calcular R² (com base no subset)
        if tipo_regressao == 'linear':
            y_pred = p(subset['Carga'])
        else:
            logRig_pred = p(subset['logQ'])
            y_pred = 10**(logRig_pred)
        y_obs = subset['rigidez']
        corr_matrix = np.corrcoef(y_pred, y_obs)
        corr = corr_matrix[0, 1]
        R_sq = corr**2

        # Plotar a regressão no intervalo global
        x_vals = np.linspace(x_min_global, x_max_global, 200)
        if tipo_regressao == 'linear':
            y_vals = p(x_vals)
        else:
            logRig_vals = p(np.log10(x_vals))
            y_vals = 10**(logRig_vals)

        plt.plot(x_vals, y_vals, color=colors[i], label=f'Regressão {num_romanos[i+1]}' if idioma == 'Português' else f'Regression {i+1}')

        # Texto no meio do subset (apenas ilustrativo)
        x_centro = (subset['Carga'].min() + subset['Carga'].max()) / 2
        if tipo_regressao == 'linear':
            y_centro = p(x_centro)
        else:
            y_centro = 10**(p(np.log10(x_centro)))
        plt.text(x_centro, y_centro*1.15, f'{num_romanos[i+1]}', color=colors[i], fontsize=20, fontweight='bold', ha='center')

        # Mostrar resultados
        if idioma == "Português":
            st.markdown(f"<b style='color:{colors[i]};'>Pontos {num_romanos[i+1]}: {lin_in} até {lin_fim}</b>", unsafe_allow_html=True)
            st.write('Tipo de regressão:', tipo_regressao.capitalize())
            st.markdown(f'<span style="color:{colors[i]};"><strong>Equação:</strong> {equacao}</span>', unsafe_allow_html=True)
            st.write('R²:', R_sq)
            quc_val = calcular_quc(reg, tipo_regressao, recalque_critico)
            st.write(f'Quc = {quc_val:.2f} tf (para recalque crítico de {recalque_critico:.2f} mm)')
        else:
            st.markdown(f"<b style='color:{colors[i]};'>Points {num_romanos[i+1]}: {lin_in} to {lin_fim}</b>", unsafe_allow_html=True)
            st.write('Regression type:', tipo_regressao.capitalize())
            st.markdown(f'<span style="color:{colors[i]};"><strong>Equation:</strong> {equacao}</span>', unsafe_allow_html=True)
            st.write('R²:', R_sq)
            quc_val = calcular_quc(reg, tipo_regressao, recalque_critico)
            st.write(f'Quc = {quc_val:.2f} tf (for critical settlement = {recalque_critico:.2f} mm)')

        # Se o usuário forneceu recalque_input > 0, calcula a carga
        if recalque_input > 0:
            carga_calculada = calcular_quc(reg, tipo_regressao, recalque_input)
            st.write(f"A carga para recalque {recalque_input:.2f} mm = {carga_calculada:.2f} tf.")

        # Se o usuário forneceu carga_input > 0, calcula o recalque
        if carga_input > 0:
            if tipo_regressao == 'linear':
                rigidez_calc = p(carga_input)
            else:
                rigidez_calc = 10**(p(np.log10(carga_input)))
            recalque_calculado = carga_input / rigidez_calc
            st.write(f"Para carga de {carga_input:.2f} tf, o recalque será {recalque_calculado:.2f} mm.")

    # 2) ACHAR INTERSEÇÕES ENTRE TODAS AS REGRESSÕES
    interseccoes = []
    for i in range(num_regressoes):
        reg_i, tipo_i = regressions[i], tipos[i]
        x_min_i, x_max_i = intervals[i]
        for j in range(i+1, num_regressoes):
            reg_j, tipo_j = regressions[j], tipos[j]
            x_min_j, x_max_j = intervals[j]
            # Sobreposição do intervalo de cada regressão
            x_min_overlap = max(x_min_i, x_min_j)
            x_max_overlap = min(x_max_i, x_max_j)
            if x_min_overlap < x_max_overlap:
                intersec = calcular_interseccao(reg_i, reg_j, tipo_i, tipo_j,
                                                x_min_overlap, x_max_overlap)
                if intersec is not None:
                    x_int, y_int = intersec
                    interseccoes.append((i, j, x_int, y_int))

    # 3) PLOTA INTERSEÇÕES COM 'X'
    for (i, j, x_int, y_int) in interseccoes:
        plt.plot(x_int, y_int, 'x', color='magenta', markersize=12, markeredgewidth=3)
        if idioma == "Português":
            st.write(f"Interseção entre regressão {num_romanos[i+1]} e {num_romanos[j+1]}: x={x_int:.2f}, y={y_int:.2f}")
        else:
            st.write(f"Intersection between regression {num_romanos[i+1]} and {num_romanos[j+1]}: x={x_int:.2f}, y={y_int:.2f}")

    # Finalizar plot
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
            st.error("Formato de coluna inválido. Verifique as colunas do arquivo.")
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

        # Inputs
        recalque_input = st.number_input(
            'Quer calcular a carga para qual recalque? (mm)' if idioma == "Português" else 'Which settlement to calculate load for? (mm)',
            format="%.2f", min_value=0.0
        )
        carga_input = st.number_input(
            'Quer estimar o recalque para qual carga? (tf)' if idioma == "Português" else 'Which load to estimate settlement for? (tf)',
            format="%.2f", min_value=0.0
        )

        # Plotly 1: Carga vs Recalque
        fig = px.scatter(
            tabela, x="Carga", y="Recalque",
            labels={"Carga": "Carga (tf)", "Recalque": "Recalque (mm)"} if idioma == "Português" else
                   {"Carga": "Load (tf)", "Recalque": "Settlement (mm)"}
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title="Carga vs Recalque" if idioma == "Português" else "Load vs Settlement",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Recalque (mm)" if idioma == "Português" else "Settlement (mm)"
        )
        for i, row in tabela.iterrows():
            fig.add_annotation(x=row["Carga"], y=row["Recalque"], text=str(i), showarrow=True, arrowhead=1, ax=20, ay=-20)
        config_plotly = {"staticPlot": True, "displayModeBar": False}
        st.plotly_chart(fig, config=config_plotly)

        # Plotly 2: Carga vs Rigidez
        fig2 = px.scatter(
            tabela, x="Carga", y="rigidez",
            labels={"Carga": "Carga (tf)", "rigidez": "Rigidez (tf/mm)"} if idioma == "Português" else
                   {"Carga": "Load (tf)", "rigidez": "Stiffness (tf/mm)"}
        )
        fig2.update_layout(
            title="Carga vs Rigidez" if idioma == "Português" else "Load vs Stiffness",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Rigidez (tf/mm)" if idioma == "Português" else "Stiffness (tf/mm)"
        )
        for i, row in tabela.iterrows():
            fig2.add_annotation(x=row["Carga"], y=row["rigidez"], text=str(i), showarrow=True, arrowhead=1, ax=20, ay=-20)
        st.plotly_chart(fig2, config=config_plotly)

        # Quantas regressões
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
                f'Ponto inicial da regressão {num_romanos[i+1]} (0-based):' if idioma == "Português" else f'Start point {num_romanos[i+1]} (0-based):',
                value=lin_in_default, key=lin_in_key
            )
            lin_fim_str = st.text_input(
                f'Ponto final da regressão {num_romanos[i+1]} (0-based):' if idioma == "Português" else f'End point {num_romanos[i+1]} (0-based):',
                value=lin_fim_default, key=lin_fim_key
            )

            try:
                lin_in_val = int(lin_in_str)
            except ValueError:
                st.error(f"Ponto inicial inválido para regressão {num_romanos[i+1]}.")
                return
            try:
                lin_fim_val = int(lin_fim_str)
            except ValueError:
                st.error(f"Ponto final inválido para regressão {num_romanos[i+1]}.")
                return

            if lin_in_val < 0 or lin_in_val >= len(tabela):
                st.error(f"Ponto inicial deve estar entre 0 e {len(tabela)-1}.")
                return
            if lin_fim_val < lin_in_val or lin_fim_val >= len(tabela):
                st.error(f"Ponto final deve estar entre {lin_in_val} e {len(tabela)-1}.")
                return

            tipo_reg_val = st.selectbox(
                f'Tipo de regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Regression type {num_romanos[i+1]}:',
                ['linear', 'log'], index=0, key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao(
                tabela, num_regressoes, pontos_tipos,
                diametro_estaca, idioma, carga_input, recalque_input
            )

# Rode o programa
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)