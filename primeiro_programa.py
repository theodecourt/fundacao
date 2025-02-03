import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq
import streamlit as st
import plotly.express as px
import io

# Dicionário para números romanos
num_romanos = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}

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
    Tenta calcular TODAS as interseções entre duas regressões (linear ou log)
    dentro de [x_min, x_max].
    Retorna uma lista de (x_int, y_int).
    """
    interseccoes = []
    
    # Funções de checagem
    def esta_no_intervalo(x):
        return (x_min <= x <= x_max)
    def esta_no_intervalo_positivo(x):
        return (x > 0) and (x_min <= x <= x_max)

    # =============== CASO 1: Ambas lineares ===============
    if tipo1 == 'linear' and tipo2 == 'linear':
        # reg = [a, b] => y = a*x + b
        a1, b1 = reg1[0], reg1[1]
        a2, b2 = reg2[0], reg2[1]

        if np.isclose(a1, a2, atol=1e-12):
            return []  # paralelas, não há ponto único

        x_int = (b2 - b1) / (a1 - a2)
        y_int = a1*x_int + b1
        if esta_no_intervalo(x_int):
            interseccoes.append((x_int, y_int))

    # =============== CASO 2: Ambas log (log-log) ===============
    elif tipo1 == 'log' and tipo2 == 'log':
        a1, b1 = reg1
        a2, b2 = reg2
        if np.isclose(a1, a2, atol=1e-12):
            return []

        # a1*log10(x) + b1 = a2*log10(x) + b2
        # => (a1 - a2)*log10(x) = b2 - b1
        log_x = (b2 - b1)/(a1 - a2)
        x_int = 10**(log_x)
        y_int = 10**(a1*log_x + b1)
        if esta_no_intervalo_positivo(x_int):
            interseccoes.append((x_int, y_int))

    # =============== CASO 3: Uma linear, outra logarítmica ===============
    else:
        # Identificar quem é linear e quem é log
        if tipo1 == 'linear':
            a_lin, b_lin = reg1  # y_lin = a_lin*x + b_lin
            a_log, b_log = reg2  # y_log = 10^(a_log * log10(x) + b_log)
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1

        def func_intersec(x):
            if x <= 0:
                return np.nan
            val_lin = a_lin*x + b_lin
            val_log = 10**(a_log*np.log10(x) + b_log)
            return val_lin - val_log

        # Varredura
        N = 300
        xx = np.linspace(x_min, x_max, N)
        ff = []
        for xi in xx:
            f_val = func_intersec(xi)
            ff.append(f_val)

        for i in range(N-1):
            f1, f2 = ff[i], ff[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1*f2 < 0:  # mudança de sinal
                try:
                    raiz = brentq(func_intersec, xx[i], xx[i+1], xtol=1e-12)
                    if esta_no_intervalo_positivo(raiz):
                        val_lin = a_lin*raiz + b_lin
                        interseccoes.append((raiz, val_lin))
                except ValueError:
                    pass

    return interseccoes

def calcular_quc(reg, tipo_regressao, valor_critico):
    """
    Exemplo de cálculo de Quc (carga) para um recalque = valor_critico,
    usando a equação rigidez = carga / recalque.
    Ajuste a equação conforme seu modelo específico.
    """
    # Para simplificar, resolvemos rigidez(x) = x / valor_critico,
    # ou seja: a*(x) + b = x/valor_critico para a regressão linear, etc.
    if tipo_regressao == 'linear':
        a, b = reg[0], reg[1]  # y = a*x + b
        def func_q(x):
            rig = a*x + b
            return rig - (x/valor_critico)
        try:
            return brentq(func_q, 1e-6, 1e9)
        except ValueError:
            return np.nan
    else:  # log
        def func_q(x):
            if x <= 0:
                return 99999  # Evitar x<=0
            rig_log = 10**(reg[0]*np.log10(x) + reg[1])
            return rig_log - (x/valor_critico)
        try:
            return brentq(func_q, 1e-6, 1e9)
        except ValueError:
            return np.nan

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    1) Ajusta cada regressão em seu subset.
    2) Calcula as interseções entre regressões consecutivas.
    3) Plota cada regressão APENAS em seu subset [x_min, x_max],
       interrompendo se houver interseção dentro desse intervalo.
    4) Desenha um "X" magenta na interseção (se cair no domínio das duas curvas).
    """

    # Ordena a tabela por Carga
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)

    # Plotar pontos originais
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Anotar índices
    for i, (xx, yy) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (xx, yy), textcoords="offset points", xytext=(0,5), ha='center')

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Guardar informações de cada regressão
    regressions = []
    tipos = []
    domains = []  # (x_min, x_max)
    
    # 1) Ajustar cada regressão e guardar
    for i in range(num_regressoes):
        lin_in = pontos_tipos[i][0]
        lin_fim = pontos_tipos[i][1]
        tipo_reg = pontos_tipos[i][2]

        subset = tabela.iloc[lin_in : lin_fim+1]
        x_sub = subset['Carga']
        y_sub = subset['rigidez']

        if tipo_reg == 'linear':
            # polyfit => [a, b], y = a*x + b
            reg = np.polyfit(x_sub, y_sub, 1)
            eq_str = f"rigidez = {reg[0]:.4f} * Carga + {reg[1]:.4f}"
        else:
            # log => y = 10^( a*log10(x) + b )
            # => polyfit(logQ, logRig, 1) => [a,b]
            reg = np.polyfit(subset['logQ'], subset['logRig'], 1)
            eq_str = f"log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}"

        # domínio real dessa regressão
        x_min = x_sub.min()
        x_max = x_sub.max()

        # Calcular R²
        if tipo_reg == 'linear':
            y_pred = np.polyval(reg, x_sub)
        else:
            # y_pred = 10^(reg[0]*log10(x) + reg[1])
            y_pred = 10**( np.polyval(reg, subset['logQ']) )
        corr = np.corrcoef(y_sub, y_pred)[0,1]
        R2 = corr**2 if not np.isnan(corr) else 0

        regressions.append(reg)
        tipos.append(tipo_reg)
        domains.append((x_min, x_max))

        # Exibir no Streamlit
        if idioma == "Português":
            st.write(f"**Regressão {num_romanos[i+1]}** - Pontos {lin_in} a {lin_fim}")
            st.write("Tipo:", tipo_reg)
            st.write("Equação:", eq_str)
            st.write("R²:", R2)
        else:
            st.write(f"**Regression {num_romanos[i+1]}** - Points {lin_in} to {lin_fim}")
            st.write("Type:", tipo_reg)
            st.write("Equation:", eq_str)
            st.write("R²:", R2)

        # Se o usuário forneceu recalque_input > 0, calcula a carga p/ esse recalque
        if recalque_input > 0:
            quc_calc = calcular_quc(reg, tipo_reg, recalque_input)
            if idioma == "Português":
                st.write(f"Carga p/ recalque {recalque_input} mm ≈ {quc_calc:.2f} tf (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"Load for settlement {recalque_input} mm ≈ {quc_calc:.2f} tf (Reg. {num_romanos[i+1]})")

        # Se o usuário forneceu carga_input > 0, calcula recalque
        if carga_input > 0:
            def rig(x):
                if tipo_reg == 'linear':
                    return np.polyval(reg, x)
                else:
                    return 10**( np.polyval(reg, np.log10(x)) ) if x>0 else np.nan
            rig_val = rig(carga_input)
            rec_val = carga_input / rig_val if rig_val>0 else np.nan
            if idioma == "Português":
                st.write(f"Recalque p/ carga {carga_input:.2f} tf ≈ {rec_val:.2f} mm (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"Settlement for load {carga_input:.2f} tf ≈ {rec_val:.2f} mm (Reg. {num_romanos[i+1]})")

    # 2) Calcular interseções consecutivas
    interseccoes = [None]*(num_regressoes-1)

    for i in range(num_regressoes-1):
        reg_i, tipo_i = regressions[i], tipos[i]
        reg_j, tipo_j = regressions[i+1], tipos[i+1]
        x_min_i, x_max_i = domains[i]
        x_min_j, x_max_j = domains[i+1]

        # Vamos considerar a UNIÃO de domínios para achar interseção
        x_min_union = min(x_min_i, x_min_j)
        x_max_union = max(x_max_i, x_max_j)

        if x_min_union < x_max_union:
            inters_ij = calcular_interseccao(
                reg_i, reg_j,
                tipo_i, tipo_j,
                x_min_union, x_max_union
            )
            if len(inters_ij) > 0:
                # Se houver mais de uma, pegamos a primeira
                # (ou poderia escolher a que efetivamente cai dentro do overlap)
                interseccoes[i] = inters_ij[0]
            else:
                interseccoes[i] = None
        else:
            interseccoes[i] = None

    # 3) Plotar cada regressão somente no seu [x_min, x_max]
    def rigidez_model(reg, tipo, x):
        """Função que, dado reg e tipo, retorna rigidez em x."""
        if tipo == 'linear':
            return np.polyval(reg, x)
        else:
            return 10**( np.polyval(reg, np.log10(x)) ) if x>0 else np.nan

    for i in range(num_regressoes):
        reg = regressions[i]
        tipo = tipos[i]
        x_min_i, x_max_i = domains[i]

        # Determinar onde começa a plotagem
        #   Se tem interseção com a anterior e ela cai no [x_min_i, x_max_i],
        #   começamos ali. Caso contrário, x_plot_start = x_min_i.
        x_plot_start = x_min_i
        if i > 0 and interseccoes[i-1] is not None:
            x_int_prev, y_int_prev = interseccoes[i-1]
            # Se a interseção cair no DOMÍNIO desta regressão:
            if x_min_i <= x_int_prev <= x_max_i:
                x_plot_start = x_int_prev

        # Determinar onde termina a plotagem
        #   Se tem interseção com a próxima e ela cai dentro do [x_min_i, x_max_i],
        #   paramos nela. Caso contrário, x_plot_end = x_max_i.
        x_plot_end = x_max_i
        if i < num_regressoes-1 and interseccoes[i] is not None:
            x_int_next, y_int_next = interseccoes[i]
            if x_min_i <= x_int_next <= x_max_i:
                x_plot_end = x_int_next

        # Gera pontos
        if x_plot_end < x_plot_start:
            # Se acontecer de a interseção ficar "antes" do start, não plota nada
            continue

        x_vals = np.linspace(x_plot_start, x_plot_end, 300)
        y_vals = [rigidez_model(reg, tipo, xv) for xv in x_vals]

        plt.plot(
            x_vals, y_vals,
            color=colors[i],
            label=f"Regressão {num_romanos[i+1]}" if idioma == "Português" else f"Regression {num_romanos[i+1]}"
        )

    # 4) Desenhar o "X" nas interseções, se estiverem dentro do domínio das duas regressões
    for i, inters in enumerate(interseccoes):
        if inters is not None:
            x_int, y_int = inters
            # Check se está no domínio da reg i e da reg i+1
            xi_min, xi_max = domains[i]
            xj_min, xj_max = domains[i+1]
            in_domain_i   = (xi_min <= x_int <= xi_max)
            in_domain_j   = (xj_min <= x_int <= xj_max)
            if in_domain_i and in_domain_j:
                plt.plot(x_int, y_int, 'x', color='magenta', markersize=12, markeredgewidth=3)
                if idioma == "Português":
                    st.write(f"Interseção Reg. {num_romanos[i+1]} e {num_romanos[i+2]}: x={x_int:.2f}, y={y_int:.2f}")
                else:
                    st.write(f"Intersection Reg. {num_romanos[i+1]} and {num_romanos[i+2]}: x={x_int:.2f}, y={y_int:.2f}")

    # Finalizar gráfico
    if idioma == "Português":
        plt.xlabel('Carga (tf)')
        plt.ylabel('Rigidez (tf/mm)')
        plt.title('Regressões de Carga vs Rigidez')
    else:
        plt.xlabel('Load (tf)')
        plt.ylabel('Stiffness (tf/mm)')
        plt.title('Load vs Stiffness Regressions')

    plt.legend(loc='best')
    st.pyplot(plt)

def primeiro_programa(idioma):
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        # Renomear colunas (para "Carga" e "Recalque")
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
            st.error("As colunas de Carga e Recalque devem conter apenas valores numéricos.")
            return

        # Calcular rigidez e logs
        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ']    = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq']  = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig']  = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        # Inputs no Streamlit
        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)',
            min_value=0.01, format="%.2f"
        )

        recalque_input = st.number_input(
            'Calcular carga para qual recalque? (mm)' if idioma == "Português" else 'Calculate load for which settlement? (mm)',
            format="%.2f", min_value=0.0
        )

        carga_input = st.number_input(
            'Estimar recalque para qual carga? (tf)' if idioma == "Português" else 'Estimate settlement for which load? (tf)',
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
            tipo_reg_key = f'tipo_regressao_{i}'

            lin_in_default = st.session_state.get(lin_in_key, '0')
            lin_fim_default = st.session_state.get(lin_fim_key, str(len(tabela)-1))
            tipo_reg_default = st.session_state.get(tipo_reg_key, 'linear')

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
                st.error("Use valores inteiros para início e fim (0-based).")
                return

            if lin_in_val < 0 or lin_in_val >= len(tabela):
                st.error(f"Ponto inicial {lin_in_val} fora do range [0, {len(tabela)-1}].")
                return
            if lin_fim_val < lin_in_val or lin_fim_val >= len(tabela):
                st.error(f"Ponto final {lin_fim_val} deve estar entre {lin_in_val} e {len(tabela)-1}.")
                return

            # Tipo de regressão
            tipo_reg_val = st.selectbox(
                (f"Tipo de regressão {num_romanos[i+1]}:" 
                 if idioma == "Português"
                 else f"Regression type {num_romanos[i+1]}:"),
                ['linear', 'log'],
                key=tipo_reg_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        if st.button("Calcular Regressões" if idioma == "Português" else "Calculate Regressions"):
            calcular_regressao(
                tabela,
                num_regressoes, 
                pontos_tipos,
                diametro_estaca,
                idioma,
                carga_input,
                recalque_input
            )

# Rode seu programa definindo o idioma desejado
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)