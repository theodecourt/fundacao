import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq
import streamlit as st
import plotly.express as px
import io

# Dicionário para números romanos
num_romanos = {
    1: 'I', 2: 'II', 3: 'III', 
    4: 'IV', 5: 'V', 6: 'VI'
}

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
        """, unsafe_allow_html=True
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
    """
    Calcula TODAS as interseções entre duas regressões (linear ou log),
    dentro do intervalo [x_min, x_max].
    Retorna lista de (x_int, y_int).
    """
    interseccoes = []
    def in_interval(x):
        return (x_min <= x <= x_max)
    
    # Caso 1: Ambas lineares
    if tipo1 == 'linear' and tipo2 == 'linear':
        a1, b1 = reg1[0], reg1[1]
        a2, b2 = reg2[0], reg2[1]
        # y1 = a1*x + b1, y2 = a2*x + b2
        if not np.isclose(a1, a2):
            x_int = (b2 - b1)/(a1 - a2)
            y_int = a1*x_int + b1
            if in_interval(x_int):
                interseccoes.append((x_int, y_int))

    # Caso 2: Ambas log
    elif tipo1 == 'log' and tipo2 == 'log':
        a1, b1 = reg1
        a2, b2 = reg2
        # y1 = 10^(a1 log10(x) + b1), y2 = 10^(a2 log10(x) + b2)
        if not np.isclose(a1, a2):
            log_x = (b2 - b1)/(a1 - a2)
            x_int = 10**(log_x)
            if x_int > 0 and in_interval(x_int):
                y_int = 10**(a1*log_x + b1)
                interseccoes.append((x_int, y_int))

    # Caso 3: Uma linear e outra log
    else:
        if tipo1 == 'linear':
            aL, bL = reg1  # y_lin = aL*x + bL
            aG, bG = reg2  # y_log = 10^(aG*log10(x) + bG)
        else:
            aL, bL = reg2
            aG, bG = reg1

        def f_inter(x):
            # linear - log
            if x <= 0:
                return np.nan
            y_lin = aL*x + bL
            y_log = 10**(aG*np.log10(x) + bG)
            return y_lin - y_log

        N = 300
        xs = np.linspace(x_min, x_max, N)
        fs = [f_inter(x) for x in xs]
        for i in range(N-1):
            f1, f2 = fs[i], fs[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1*f2 < 0:  # há mudança de sinal
                try:
                    x_raiz = brentq(f_inter, xs[i], xs[i+1], xtol=1e-12)
                    if x_raiz > 0 and in_interval(x_raiz):
                        # y em qualquer eq
                        y_raiz = aL*x_raiz + bL
                        interseccoes.append((x_raiz, y_raiz))
                except:
                    pass
    return interseccoes

def calcular_quc(reg, tipo, rec):
    """
    Exemplo de cálculo de 'carga' para um recalque = rec,
    resolvendo rigidez(x) = x / rec.
    """
    if tipo == 'linear':
        a, b = reg[0], reg[1]
        def f_q(x):
            rig = a*x + b
            return rig - (x/rec)
        try:
            return brentq(f_q, 1e-6, 1e9)
        except ValueError:
            return np.nan
    else:
        def f_q(x):
            if x<=0:
                return 1e12
            rig = 10**( reg[0]*np.log10(x) + reg[1] )
            return rig - (x/rec)
        try:
            return brentq(f_q, 1e-6, 1e9)
        except ValueError:
            return np.nan

def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    1) Faz o ajuste de cada regressão no subset escolhido,
    2) Mas para plotar, considera o domínio global [C_min, C_max],
       'emendando' com as interseções para que cada regressão só vá
       até a interseção com a próxima (ou até o fim) e comece na
       interseção com a anterior (ou no início).
    3) Desenha 'X' na interseção.
    """

    # Ordena
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)

    # Plotar dados
    x_all = tabela['Carga']
    y_all = tabela['rigidez']

    plt.figure(figsize=(10, 6))
    plt.plot(x_all, y_all, 'go', label='Dados Originais' if idioma=='Português' else 'Original Data')
    for i, (xx, yy) in enumerate(zip(x_all, y_all)):
        plt.annotate(str(i), (xx, yy), textcoords="offset points", xytext=(0,5), ha='center')

    # Domínio global
    c_min = x_all.min()
    c_max = x_all.max()

    # Ajustar cada regressão
    regs = []
    tipos = []
    for i in range(num_regressoes):
        lin_in, lin_fim, t_reg = pontos_tipos[i]
        df_sub = tabela.iloc[lin_in : lin_fim+1]

        if t_reg=='linear':
            # regressão linear
            coefs = np.polyfit(df_sub['Carga'], df_sub['rigidez'], 1)
            eq_str = f"rig = {coefs[0]:.4f}*x + {coefs[1]:.4f}"
        else:
            # regressão log
            coefs = np.polyfit(df_sub['logQ'], df_sub['logRig'], 1)
            eq_str = f"log(rig) = {coefs[0]:.4f}*log(x) + {coefs[1]:.4f}"

        # Calcular R² no subset
        if t_reg=='linear':
            y_pred = np.polyval(coefs, df_sub['Carga'])
        else:
            logRig_pred = np.polyval(coefs, df_sub['logQ'])
            y_pred = 10**(logRig_pred)
        corr = np.corrcoef(y_pred, df_sub['rigidez'])[0,1]
        R2 = corr**2 if not np.isnan(corr) else 0

        if idioma=='Português':
            st.write(f"**Regressão {num_romanos[i+1]}**: pontos {lin_in} a {lin_fim}")
            st.write("Tipo:", t_reg)
            st.write("Equação:", eq_str)
            st.write("R²:", R2)
        else:
            st.write(f"**Regression {num_romanos[i+1]}**: points {lin_in} to {lin_fim}")
            st.write("Type:", t_reg)
            st.write("Equation:", eq_str)
            st.write("R²:", R2)

        # Se pedido, calcula Quc p/ certo recalque
        if recalque_input>0:
            quc_val = calcular_quc(coefs, t_reg, recalque_input)
            if idioma=='Português':
                st.write(f"Carga para recalque {recalque_input:.2f} mm = {quc_val:.2f} tf (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"Load for settlement {recalque_input:.2f} mm = {quc_val:.2f} tf (Reg. {num_romanos[i+1]})")

        # Se pedido, calcula recalque p/ certa carga
        if carga_input>0:
            def rigidez_model(x):
                if t_reg=='linear':
                    return np.polyval(coefs, x)
                else:
                    if x<=0:
                        return np.nan
                    return 10**( np.polyval(coefs, np.log10(x)) )
            rig_val = rigidez_model(carga_input)
            rec_val = carga_input/rig_val if rig_val>0 else np.nan
            if idioma=='Português':
                st.write(f"Recalque p/ carga {carga_input:.2f} tf = {rec_val:.2f} mm (Reg. {num_romanos[i+1]})")
            else:
                st.write(f"Settlement for load {carga_input:.2f} tf = {rec_val:.2f} mm (Reg. {num_romanos[i+1]})")

        regs.append(coefs)
        tipos.append(t_reg)

    # Calcular interseções entre regressões consecutivas, mas no domínio global
    inters_list = [None]*(num_regressoes-1)
    for i in range(num_regressoes-1):
        r1, t1 = regs[i], tipos[i]
        r2, t2 = regs[i+1], tipos[i+1]
        inters = calcular_interseccao(r1, r2, t1, t2, c_min, c_max)
        if len(inters)>0:
            # Pega a primeira (ou poderia pegar a que estiver mais no meio)
            inters_list[i] = inters[0]
        else:
            inters_list[i] = None

    # Função para calcular rigidez a partir de reg/tipo
    def rigidez_model(reg, tipo, x):
        if tipo=='linear':
            return np.polyval(reg, x)
        else:
            if x<=0:
                return np.nan
            return 10**( np.polyval(reg, np.log10(x)) )

    # Plotar cada regressão desde:
    # - a interseção com a anterior (se cair em [c_min, c_max]) 
    #   ou c_min (se i=0),
    # até:
    # - a interseção com a próxima (se existir),
    #   senão c_max.
    for i in range(num_regressoes):
        x_start = c_min if i==0 else c_min
        # se houver interseção com anterior:
        if i>0 and inters_list[i-1] is not None:
            x_prev, y_prev = inters_list[i-1]
            # se estiver dentro [c_min, c_max], começa dali
            if c_min <= x_prev <= c_max:
                x_start = x_prev

        x_end = c_max
        if i< num_regressoes-1 and inters_list[i] is not None:
            x_next, y_next = inters_list[i]
            if c_min <= x_next <= c_max:
                x_end = x_next

        if x_end < x_start:
            continue

        xx = np.linspace(x_start, x_end, 300)
        yy = [rigidez_model(regs[i], tipos[i], x) for x in xx]
        plt.plot(xx, yy, label=f"Reg. {num_romanos[i+1]}", color=plt.cm.tab10(i))

    # Marcar interseções
    for i, inters in enumerate(inters_list):
        if inters is not None:
            x_i, y_i = inters
            if c_min <= x_i <= c_max:
                plt.plot(x_i, y_i, 'x', color='magenta', markersize=12, markeredgewidth=3)
                if idioma=='Português':
                    st.write(f"Interseção Reg. {num_romanos[i+1]} e {num_romanos[i+2]}: x={x_i:.2f}, y={y_i:.2f}")
                else:
                    st.write(f"Intersection Reg. {num_romanos[i+1]} and {num_romanos[i+2]}: x={x_i:.2f}, y={y_i:.2f}")

    # Final
    if idioma=='Português':
        plt.xlabel("Carga (tf)")
        plt.ylabel("Rigidez (tf/mm)")
        plt.title("Regressões de Carga vs Rigidez")
    else:
        plt.xlabel("Load (tf)")
        plt.ylabel("Stiffness (tf/mm)")
        plt.title("Load vs Stiffness Regressions")

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
            st.error(
                "Formato de coluna inválido. "
                "Certifique-se de que o arquivo contém "
                "'Carga (tf)' e 'Recalque (mm)' ou "
                "'Load (tf)' e 'Settlement (mm)'."
            )
            return

        # Converter para numérico
        try:
            tabela['Carga'] = tabela['Carga'].astype(float)
            tabela['Recalque'] = tabela['Recalque'].astype(float)
        except ValueError:
            st.error("As colunas de Carga e Recalque devem conter apenas valores numéricos.")
            return

        # Calcular rigidez e logs
        tabela['rigidez'] = tabela['Carga']/tabela['Recalque']
        tabela['logQ']    = tabela['Carga'].apply(lambda x: math.log10(x) if x>0 else np.nan)
        tabela['logReq']  = tabela['Recalque'].apply(lambda x: math.log10(x) if x>0 else np.nan)
        tabela['logRig']  = tabela['rigidez'].apply(lambda x: math.log10(x) if x>0 else np.nan)

        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma=="Português" else 'What is the pile diameter? (mm)',
            min_value=0.01, format="%.2f"
        )
        recalque_input = st.number_input(
            'Calcular carga p/ qual recalque? (mm)' if idioma=="Português" else 'Calculate load for which settlement? (mm)',
            min_value=0.0, format="%.2f"
        )
        carga_input = st.number_input(
            'Estimar recalque p/ qual carga? (tf)' if idioma=="Português" else 'Estimate settlement for which load? (tf)',
            min_value=0.0, format="%.2f"
        )

        # Plotly - Carga x Recalque
        fig = px.scatter(
            tabela, x="Carga", y="Recalque",
            labels=(
                {"Carga":"Carga (tf)", "Recalque":"Recalque (mm)"}
                if idioma=="Português" else
                {"Carga":"Load (tf)",  "Recalque":"Settlement (mm)"}
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            title=("Carga vs Recalque" if idioma=="Português" else "Load vs Settlement"),
            xaxis_title=("Carga (tf)" if idioma=="Português" else "Load (tf)"),
            yaxis_title=("Recalque (mm)" if idioma=="Português" else "Settlement (mm)")
        )
        for i, row in tabela.iterrows():
            fig.add_annotation(
                x=row['Carga'], y=row['Recalque'],
                text=str(i), showarrow=True, arrowhead=1, ax=20, ay=-20
            )
        st.plotly_chart(fig, use_container_width=True)

        # Plotly - Carga x Rigidez
        fig2 = px.scatter(
            tabela, x="Carga", y="rigidez",
            labels=(
                {"Carga":"Carga (tf)", "rigidez":"Rigidez (tf/mm)"}
                if idioma=="Português" else
                {"Carga":"Load (tf)",  "rigidez":"Stiffness (tf/mm)"}
            )
        )
        fig2.update_layout(
            title=("Carga vs Rigidez" if idioma=="Português" else "Load vs Stiffness"),
            xaxis_title=("Carga (tf)" if idioma=="Português" else "Load (tf)"),
            yaxis_title=("Rigidez (tf/mm)" if idioma=="Português" else "Stiffness (tf/mm)")
        )
        for i, row in tabela.iterrows():
            fig2.add_annotation(
                x=row['Carga'], y=row['rigidez'],
                text=str(i), showarrow=True, arrowhead=1, ax=20, ay=-20
            )
        st.plotly_chart(fig2, use_container_width=True)

        # Escolha de quantas regressões
        num_reg = st.selectbox(
            'Quantas regressões?' if idioma=="Português" else "How many regressions?",
            [1,2,3,4,5],
            index=0
        )

        pontos_tipos = []
        for i in range(num_reg):
            lin_in_key = f'lin_in_{i}'
            lin_fim_key = f'lin_fim_{i}'
            tipo_reg_key = f'tipo_regressao_{i}'

            lin_in_def = st.session_state.get(lin_in_key, '0')
            lin_fim_def = st.session_state.get(lin_fim_key, str(len(tabela)-1))
            tipo_reg_def = st.session_state.get(tipo_reg_key, 'linear')

            # Input
            lin_in_str = st.text_input(
                (f"Ponto inicial da regressão {num_romanos[i+1]} (0-based):" 
                 if idioma=="Português" else
                 f"Starting point of regression {num_romanos[i+1]} (0-based):"),
                value=lin_in_def, key=lin_in_key
            )
            lin_fim_str = st.text_input(
                (f"Ponto final da regressão {num_romanos[i+1]} (0-based):" 
                 if idioma=="Português" else
                 f"Ending point of regression {num_romanos[i+1]} (0-based):"),
                value=lin_fim_def, key=lin_fim_key
            )

            try:
                lin_in_val = int(lin_in_str)
                lin_fim_val = int(lin_fim_str)
            except ValueError:
                st.error("Use valores inteiros para os índices.")
                return

            if lin_in_val<0 or lin_in_val>=len(tabela):
                st.error(f"Início deve estar entre 0 e {len(tabela)-1}.")
                return
            if lin_fim_val<lin_in_val or lin_fim_val>=len(tabela):
                st.error(f"Fim deve estar entre {lin_in_val} e {len(tabela)-1}.")
                return

            # Tipo
            tipo_reg_val = st.selectbox(
                (f"Tipo de regressão {num_romanos[i+1]}:" 
                 if idioma=="Português" else
                 f"Regression type {num_romanos[i+1]}:"),
                ['linear','log'], key=tipo_reg_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        if st.button("Calcular Regressões" if idioma=="Português" else "Calculate Regressions"):
            calcular_regressao(
                tabela, num_reg, pontos_tipos,
                diametro_estaca, idioma,
                carga_input, recalque_input
            )

# Executar
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)