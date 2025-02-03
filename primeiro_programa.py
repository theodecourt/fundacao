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
    """
    Tenta calcular TODAS as interseções entre duas regressões (linear ou log),
    se existirem, dentro do intervalo [x_min, x_max].
    Retorna uma lista de (x_int, y_int) dentro desse intervalo.
    """
    interseccoes = []
    
    def esta_no_intervalo(x):
        return (x_min <= x <= x_max)
    
    def esta_no_intervalo_positivo(x):
        return (x > 0) and (x_min <= x <= x_max)

    # =============== CASO 1: Ambas lineares ===============
    if tipo1 == 'linear' and tipo2 == 'linear':
        # reg = np.polyfit(x, y) => reg[0] = a, reg[1] = b => y = a*x + b
        a1, b1 = reg1[0], reg1[1]
        a2, b2 = reg2[0], reg2[1]

        if np.isclose(a1, a2, atol=1e-12):
            return []  # paralelas (sem ponto único)

        # a1*x + b1 = a2*x + b2  =>  (a1 - a2)*x = b2 - b1
        x_int = (b2 - b1) / (a1 - a2)
        y_int = a1*x_int + b1

        if esta_no_intervalo(x_int):
            interseccoes.append([x_int, y_int])

    # =============== CASO 2: Ambas logarítmicas (log-log) ===============
    elif tipo1 == 'log' and tipo2 == 'log':
        a1, b1 = reg1
        a2, b2 = reg2

        if np.isclose(a1, a2, atol=1e-12):
            return []

        # log(rig1) = a1*log10(x) + b1
        # log(rig2) = a2*log10(x) + b2
        # => a1*log10(x) + b1 = a2*log10(x) + b2
        # => (a1 - a2)*log10(x) = b2 - b1
        # => log10(x) = (b2 - b1)/(a1 - a2)
        log_x = (b2 - b1) / (a1 - a2)
        x_int = 10**(log_x)
        y_int = 10**(a1*log_x + b1)

        if esta_no_intervalo_positivo(x_int):
            interseccoes.append([x_int, y_int])

    # =============== CASO 3: Linear + Log ===============
    else:
        # Identificar quem é linear e quem é log
        # Lembrando: se reg = [a, b] linear => y = a*x + b
        # Se reg = [a, b] log => y = 10^( a*log10(x) + b )
        if tipo1 == 'linear':
            a_lin, b_lin = reg1  # y_lin = a_lin*x + b_lin
            a_log, b_log = reg2  # y_log = 10^(a_log*log10(x) + b_log)
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1
        
        def func_intersec(x):
            val_lin = a_lin*x + b_lin
            if x <= 0:
                return np.nan
            val_log = 10**(a_log * np.log10(x) + b_log)
            return val_lin - val_log
        
        # Vamos fazer uma varredura e ver onde ocorre mudança de sinal
        num_steps = 300
        xs = np.linspace(x_min, x_max, num_steps)
        fs = []
        for xi in xs:
            fs.append(func_intersec(xi) if xi > 0 else np.nan)
        
        for i in range(len(xs)-1):
            f1, f2 = fs[i], fs[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 * f2 < 0:  # mudança de sinal => raiz no meio
                try:
                    raiz = brentq(func_intersec, xs[i], xs[i+1], xtol=1e-8)
                    if raiz > 0 and esta_no_intervalo(raiz):
                        val_lin = a_lin*raiz + b_lin
                        interseccoes.append([raiz, val_lin])
                except ValueError:
                    pass

    return interseccoes

def calcular_quc(reg, tipo_regressao, valor_critico):
    """Cálculo de Quc (carga para um determinado recalque) dependendo do tipo de regressão.
       (Exemplo: Quc para Recalque = 0.1 * D)"""
    if tipo_regressao == 'linear':
        # reg = [a, b] => y = a*x + b  => "rigidez = a*carga + b"
        # Para rigidez = carga / recalque => rigidez = a*x + b => (carga / recalque) = a*carga + b
        # => 1/recalque = a + b/carga => etc. (esta formula depende muito da modelagem)
        # Se você tem outro modelo, adapte aqui.
        # Abaixo é só ilustrativo
        a = reg[0]
        b = reg[1]
        # O "rigidez = a*x + b" => x/c = a*x + b => ...
        # Muitas vezes basta resolver numericamente. Aqui mantemos seu approach ou um brentq:
        def func_quc(x):
            rig = a*x + b  # rigidez
            return rig - (x / valor_critico)
        try:
            quc = brentq(func_quc, 1e-6, 1e9)
        except ValueError:
            quc = np.nan

    else:  # log
        # y = 10^( a*log10(x) + b ) => rigidez
        def func_quc_log(x):
            rig_log = 10**(reg[0]*np.log10(x) + reg[1])
            return rig_log - (x / valor_critico)
        try:
            quc = brentq(func_quc_log, 1e-6, 1e9)
        except ValueError:
            quc = np.nan
    return quc

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    1) Ajusta regressões para cada subset.
    2) Encontra interseções consecutivas (se existirem) dentro da sobreposição de domínios.
    3) Plota cada regressão no seu domínio, interrompendo no ponto de interseção, 
       e marca um X ali.
    """

    # Ordena a tabela por Carga
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)

    # Primeiro, vamos plotar os pontos originais (Carga x rigidez)
    x0 = tabela['Carga']
    y0 = tabela['rigidez']

    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')
    for i, (xx, yy) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (xx, yy), textcoords="offset points", xytext=(0,5), ha='center')

    colors = ['blue', 'red', 'green', 'orange', 'purple']  # se quiser mais regressões

    # Guardaremos as informações de cada regressão
    regressions = []
    tipos = []
    domains = []  # (x_min, x_max) de cada subset

    # Ajustar cada regressão e guardar
    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_regressao = pontos_tipos[i][2]

        subset = tabela.iloc[lin_in:lin_fim+1]
        x_sub = subset['Carga']
        y_sub = subset['rigidez']

        # Ajuste
        if tipo_regressao == 'linear':
            # np.polyfit(x_sub, y_sub, deg=1) => [a, b], ou seja rig = a*x + b
            reg = np.polyfit(x_sub, y_sub, 1)  
            equacao = f"rigidez = {reg[0]:.4f} * Carga + {reg[1]:.4f}"
        else:
            # log => y = 10^(a*log10(x) + b)
            # Ajuste: polyfit( logQ, logRig, 1 ) => [a, b]
            reg = np.polyfit(subset['logQ'], subset['logRig'], 1)
            equacao = f"log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}"

        # Guardar domínios (mínimo e máximo de "Carga" naquele subset)
        x_min = x_sub.min()
        x_max = x_sub.max()

        regressions.append(reg)
        tipos.append(tipo_regressao)
        domains.append((x_min, x_max))

        # Calcular R²
        if tipo_regressao == 'linear':
            y_pred = np.polyval(reg, x_sub)
        else:
            # y_pred = 10^( reg[0]*log10(x) + reg[1] )
            y_pred = 10**( np.polyval(reg, subset['logQ']) )
        corr_matrix = np.corrcoef(y_pred, y_sub)
        corr = corr_matrix[0,1]
        R_sq = corr**2

        # Exibir no Streamlit
        if idioma == "Português":
            st.markdown(f"**Regressão {num_romanos[i+1]}** - Pontos: {lin_in} até {lin_fim}")
            st.write("Tipo de regressão:", tipo_regressao)
            st.write("Equação:", equacao)
            st.write("R²:", R_sq)
        else:
            st.markdown(f"**Regression {num_romanos[i+1]}** - Points: {lin_in} to {lin_fim}")
            st.write("Regression type:", tipo_regressao)
            st.write("Equation:", equacao)
            st.write("R²:", R_sq)

        # Se o usuário forneceu recalque_input > 0, calcula a carga
        if recalque_input > 0:
            quc_val = calcular_quc(reg, tipo_regressao, recalque_input)
            if idioma == "Português":
                st.write(f"A carga para o recalque {recalque_input:.2f} mm é ~ {quc_val:.2f} tf (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"The load for settlement {recalque_input:.2f} mm is ~ {quc_val:.2f} tf (Reg. {num_romanos[i+1]})")

        # Se o usuário forneceu carga_input > 0, calcula o recalque
        if carga_input > 0:
            def rigidez_model(x):
                if tipo_regressao == 'linear':
                    return np.polyval(reg, x)
                else:
                    return 10**( np.polyval(reg, np.log10(x)) )
            rig_i = rigidez_model(carga_input)
            recalc_i = carga_input / rig_i
            if idioma == "Português":
                st.write(f"Para a carga {carga_input:.2f} tf, o recalque é ~ {recalc_i:.2f} mm (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"For load {carga_input:.2f} tf, settlement is ~ {recalc_i:.2f} mm (Reg. {num_romanos[i+1]})")

    # -------------------------------
    # Calcular interseções consecutivas
    # -------------------------------
    interseccoes = [None]*(num_regressoes-1)

    for i in range(num_regressoes-1):
        # Domínio em comum: [x_min_common, x_max_common]
        x_min_i, x_max_i = domains[i]
        x_min_j, x_max_j = domains[i+1]

        # Sobreposição
        x_min_common = max(x_min_i, x_min_j)
        x_max_common = min(x_max_i, x_max_j)

        if x_min_common < x_max_common:
            # Calcular interseções
            inters = calcular_interseccao(
                regressions[i], regressions[i+1],
                tipos[i], tipos[i+1],
                x_min_common, x_max_common
            )
            if len(inters) > 0:
                # Para simplificar, vamos considerar apenas a primeira
                interseccoes[i] = inters[0]  # (x_int, y_int)
        else:
            interseccoes[i] = None

    # -------------------------------
    # Agora fazemos a plotagem
    # -------------------------------
    for i in range(num_regressoes):
        reg = regressions[i]
        tipo = tipos[i]
        x_dom_min, x_dom_max = domains[i]

        # Definir função "rigidez_model" p/ qualquer x
        def rigidez_model(x):
            if tipo == 'linear':
                return np.polyval(reg, x)
            else:
                return 10**( np.polyval(reg, np.log10(x)) )

        # Vamos ver se há interseção com (i+1) e (i-1) 
        # para eventualmente ajustar o x_in e x_fim.
        # Mas aqui, para simplificar, faremos “cada regressão 
        # apenas no seu subset” e, se houver interseção 
        # DENTRO desse subset, interrompemos no X.

        # x_inicial:
        #   - se i > 0, e existe interseccao[i-1], e ela cai no [x_dom_min, x_dom_max], 
        #     então começamos no ponto de interseção. 
        #   - caso contrário, começamos em x_dom_min
        x_plot_start = x_dom_min
        if i > 0 and interseccoes[i-1] is not None:
            x_int_prev = interseccoes[i-1][0]
            if x_dom_min <= x_int_prev <= x_dom_max:
                x_plot_start = x_int_prev

        # x_final:
        #   - se i < num_regressoes-1, e existe interseccao[i], 
        #     e ela cai no [x_dom_min, x_dom_max],
        #     então terminamos no ponto de interseção
        #   - caso contrário, vamos até x_dom_max
        x_plot_end = x_dom_max
        if i < num_regressoes-1 and interseccoes[i] is not None:
            x_int_next = interseccoes[i][0]
            if x_dom_min <= x_int_next <= x_dom_max:
                x_plot_end = x_int_next

        # Gera pontos para plotar
        x_vals = np.linspace(x_plot_start, x_plot_end, 300)
        y_vals = rigidez_model(x_vals)

        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            label=f"Regressão {num_romanos[i+1]}" if idioma=="Português" else f"Regression {num_romanos[i+1]}"
        )

    # Marcar as interseções com 'X'
    # interseccoes[i] => intersec. da reg i com i+1
    for i, inters in enumerate(interseccoes):
        if inters is not None:
            x_int, y_int = inters
            plt.plot(
                x_int, y_int,
                'x', color='magenta', markersize=12, markeredgewidth=3
            )
            if idioma == "Português":
                st.write(f"Interseção entre Reg. {num_romanos[i+1]} e Reg. {num_romanos[i+2]}: x={x_int:.2f}, y={y_int:.2f}")
            else:
                st.write(f"Intersection between Reg. {num_romanos[i+1]} and Reg. {num_romanos[i+2]}: x={x_int:.2f}, y={y_int:.2f}")

    # Configurar e mostrar o gráfico
    if idioma == "Português":
        plt.xlabel('Carga (tf)')
        plt.ylabel('Rigidez (tf/mm)')
        plt.title('Regressões de Carga vs. Rigidez')
    else:
        plt.xlabel('Load (tf)')
        plt.ylabel('Stiffness (tf/mm)')
        plt.title('Load vs. Stiffness Regressions')

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
        
        # Converter para numérico
        try:
            tabela['Carga'] = tabela['Carga'].astype(float)
            tabela['Recalque'] = tabela['Recalque'].astype(float)
        except ValueError:
            st.error("As colunas 'Carga' e 'Recalque' devem conter apenas valores numéricos.")
            return

        # Calcular rigidez e logs
        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ']    = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq']  = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig']  = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)',
            min_value=0.01, format="%.2f"
        )

        recalque_input = st.number_input(
            'Quer calcular a carga para qual recalque? (mm)' if idioma == "Português" else 'Which settlement do you want to calculate load for? (mm)',
            format="%.2f", min_value=0.0
        )
        carga_input = st.number_input(
            'Quer estimar o recalque para qual carga? (tf)' if idioma == "Português" else 'Which load do you want to estimate settlement for? (tf)',
            format="%.2f", min_value=0.0
        )

        # Plot inicial (Plotly) - Carga vs Recalque
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
        st.plotly_chart(fig, use_container_width=True)

        # Plot inicial (Plotly) - Carga vs Rigidez
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
        st.plotly_chart(fig2, use_container_width=True)

        # Quantas regressões
        num_regressoes = st.selectbox(
            'Quantas regressões?' if idioma == "Português" else 'How many regressions?',
            [1,2,3,4],
            index=0
        )

        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in_key = f'lin_in_{i}'
            lin_fim_key = f'lin_fim_{i}'
            tipo_regressao_key = f'tipo_regressao_{i}'

            lin_in_default = st.session_state.get(lin_in_key, '0')
            lin_fim_default = st.session_state.get(lin_fim_key, str(len(tabela) - 1))
            tipo_regressao_default = st.session_state.get(tipo_regressao_key, 'linear')

            # Ponto inicial
            lin_in_str = st.text_input(
                (f"Ponto inicial da regressão {num_romanos[i+1]} (0-based):" 
                 if idioma == "Português" 
                 else f"Starting point of regression {num_romanos[i+1]} (0-based):"),
                value=lin_in_default,
                key=lin_in_key
            )

            # Ponto final
            lin_fim_str = st.text_input(
                (f"Ponto final da regressão {num_romanos[i+1]} (0-based):" 
                 if idioma == "Português" 
                 else f"Ending point of regression {num_romanos[i+1]} (0-based):"),
                value=lin_fim_default,
                key=lin_fim_key
            )

            try:
                lin_in_val = int(lin_in_str)
                lin_fim_val = int(lin_fim_str)
            except ValueError:
                st.error("Entre com valores inteiros para início e fim (0-based).")
                return

            if lin_in_val < 0 or lin_in_val >= len(tabela):
                st.error(f"Ponto inicial da regressão {num_romanos[i+1]} fora do range [0, {len(tabela)-1}].")
                return
            if lin_fim_val < lin_in_val or lin_fim_val >= len(tabela):
                st.error(f"Ponto final da regressão {num_romanos[i+1]} deve estar entre {lin_in_val} e {len(tabela)-1}.")
                return

            # Tipo de regressão
            tipo_reg_val = st.selectbox(
                (f"Tipo de regressão {num_romanos[i+1]}:" 
                 if idioma == "Português"
                 else f"Regression type {num_romanos[i+1]}:"),
                ['linear','log'],
                key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        if st.button("Calcular Regressões" if idioma == "Português" else "Calculate Regressions"):
            calcular_regressao(
                tabela, num_regressoes, pontos_tipos,
                diametro_estaca, idioma,
                carga_input, recalque_input
            )

# Executar
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)