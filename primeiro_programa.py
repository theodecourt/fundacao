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

def calcular_interseccoes(reg1, reg2, tipo1, tipo2, x_min, x_max):
    """
    Retorna uma lista de TODAS as interseções (x_int, y_int) dentro de [x_min, x_max]
    entre duas regressões definidas por (tipo, coef).
    """
    inters = []
    def in_range(x):
        return (x_min <= x <= x_max)

    def in_range_positive(x):
        return (x > 0) and (x_min <= x <= x_max)

    # -------- 1) LINEAR x LINEAR --------
    if tipo1 == 'linear' and tipo2 == 'linear':
        # reg = [a, b], com y = a*x + b
        a1, b1 = reg1
        a2, b2 = reg2
        if not np.isclose(a1, a2, atol=1e-12):
            x_int = (b2 - b1)/(a1 - a2)
            y_int = a1*x_int + b1
            if in_range(x_int):
                inters.append((x_int, y_int))

    # -------- 2) LOG x LOG --------
    elif tipo1 == 'log' and tipo2 == 'log':
        a1, b1 = reg1
        a2, b2 = reg2
        if not np.isclose(a1, a2, atol=1e-12):
            log_x = (b2 - b1)/(a1 - a2)
            x_int = 10**(log_x)
            if in_range_positive(x_int):
                y_int = 10**(a1*log_x + b1)
                inters.append((x_int, y_int))

    # -------- 3) LINEAR x LOG --------
    else:
        if tipo1 == 'linear':
            a_lin, b_lin = reg1  # y_lin = a_lin*x + b_lin
            a_log, b_log = reg2  # y_log = 10^(a_log * log10(x) + b_log)
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1
        
        def func(x):
            if x <= 0:
                return np.nan
            y_lin = a_lin*x + b_lin
            y_log = 10**(a_log*np.log10(x) + b_log)
            return y_lin - y_log

        # Discretiza e acha as faixas de mudança de sinal
        N = 200
        xs = np.linspace(x_min, x_max, N)
        fvals = [func(xx) for xx in xs]

        for i in range(N-1):
            f1, f2 = fvals[i], fvals[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1*f2 < 0:  # mudança de sinal => raiz no meio
                try:
                    raiz = brentq(func, xs[i], xs[i+1])
                    if raiz > 0 and in_range(raiz):
                        y_int = a_lin*raiz + b_lin  # (ou do log, dá na mesma)
                        inters.append((raiz, y_int))
                except ValueError:
                    pass

    return inters

def calcular_quc(reg, tipo_regressao, valor_critico):
    """Cálculo de Quc (carga para um determinado recalque) dependendo do tipo de regressão."""
    if tipo_regressao == 'linear':
        # reg = [a, b] => y = a*x + b
        a, b = reg
        # No seu código original, havia outra fórmula. Mas vou deixar coerente:
        #   rigidez = a*x + b
        #   rigidez = carga/recalque => (carga/recalque) = a*carga + b
        #   ...
        # Se preferir, mantenha a sua forma antiga. Aqui, vou usar uma brentq:
        def f_q(q):
            y_calc = a*q + b  # rigidez
            return y_calc - q/valor_critico
        try:
            quc = brentq(f_q, 1e-6, 1e9)
        except ValueError:
            quc = np.nan
        return quc
    else:  # 'log'
        a, b = reg
        # rigidez = 10^(a*log10(q) + b)
        def f_q(q):
            if q <= 0:
                return np.nan
            rig_calc = 10**(a*np.log10(q) + b)
            return rig_calc - q/valor_critico
        try:
            quc = brentq(f_q, 1e-6, 1e9)
        except ValueError:
            quc = np.nan
        return quc

def plot_regressoes(tabela, info_regressoes, inters_adj, idioma, carga_input, recalque_input, diametro_estaca):
    """
    - info_regressoes: lista de dicionários com:
        {
          'tipo': 'linear'/'log',
          'coef': [a, b]  (saída do polyfit),
          'x_min': float,
          'x_max': float,
          'lin_in': int,
          'lin_fim': int
        }
    - inters_adj: lista de interseções entre cada i e i+1 (ou None).
      ex: inters_adj[i] = (x_int, y_int) => interseção entre regressão i e i+1
    """
    x0 = tabela['Carga'].values
    y0 = tabela['rigidez'].values

    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Anotar índices no gráfico
    for i, (xx, yy) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (xx, yy), textcoords="offset points", xytext=(0,5), ha='center')

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    recalque_critico = 0.1 * diametro_estaca

    for i, reginfo in enumerate(info_regressoes):
        tipo_reg = reginfo['tipo']
        reg_coef = reginfo['coef']
        a, b = reg_coef  # (só para exibir a equação ou algo assim)
        x_min_i = reginfo['x_min']
        x_max_i = reginfo['x_max']

        # ---- Descobrir limites de plot (x_start, x_end) ----
        #   - Se existe interseção com i-1 => esse é o x_start
        #   - Se existe interseção com i+1 => esse é o x_end
        # Caso não exista ou esteja fora do domínio, usa x_min_i ou x_max_i.

        # (x_start) => interseção com i-1
        if i == 0:
            x_start = x_min_i
        else:
            # inters_adj[i-1] é a interseção entre (i-1) e i
            intersec_esq = inters_adj[i-1]  # (x_int, y_int) ou None
            if intersec_esq is not None:
                x_int_esq = intersec_esq[0]
                # tem que estar dentro do domínio i e também do i-1
                if x_min_i <= x_int_esq <= x_max_i:
                    x_start = x_int_esq
                else:
                    x_start = x_min_i
            else:
                x_start = x_min_i

        # (x_end) => interseção com i+1
        if i == len(info_regressoes) - 1:
            x_end = x_max_i
        else:
            intersec_dir = inters_adj[i]  # (x_int, y_int) ou None, c/ a regressão i+1
            if intersec_dir is not None:
                x_int_dir = intersec_dir[0]
                if x_min_i <= x_int_dir <= x_max_i:
                    x_end = x_int_dir
                else:
                    x_end = x_max_i
            else:
                x_end = x_max_i

        # Precisamos garantir x_start < x_end. Se der ao contrário, não plota nada
        if x_start >= x_end:
            # Pode acontecer se a interseção estiver fora do domínio
            continue

        # Montar função de rigidez
        def rigidez_func(x):
            if tipo_reg == 'linear':
                return a*x + b
            else:
                # log => rig = 10^(a * log10(x) + b)
                return 10**(a*np.log10(x) + b)

        # Gera uma curva suave no intervalo [x_start, x_end]
        xx_plot = np.linspace(x_start, x_end, 200)
        yy_plot = [rigidez_func(xx) for xx in xx_plot]

        # Plot
        plt.plot(xx_plot, yy_plot, color=colors[i], label=f'Regressão {i+1}' if idioma == 'Português' else f'Regression {i+1}')

        # Adicionar um texto com o número romano no meio
        mid_index = len(xx_plot)//2
        xm, ym = xx_plot[mid_index], yy_plot[mid_index]
        plt.text(
            xm, ym*1.03,
            f'{num_romanos[i+1]}',
            color=colors[i],
            fontsize=14, fontweight='bold', ha='center'
        )

        # Exibir no Streamlit algumas informações
        if tipo_reg == 'linear':
            eq_str = f'rigidez = {a:.4f} * x + {b:.4f}'
        else:
            eq_str = f'log(rigidez) = {a:.4f} * log(x) + {b:.4f}'

        # R² (vamos usar a correlação do subset original do ajuste)
        #   Se quisermos o R² linear, podemos verificar reginfo['R2'] se armazenamos
        st.write(f"---\n**Regressão {num_romanos[i+1]}**")
        st.write(f"Pontos usados: {reginfo['lin_in']} até {reginfo['lin_fim']}")
        st.write(f"Tipo: {tipo_reg}")
        st.write(f"Equação: {eq_str}")
        if 'R2' in reginfo:
            st.write(f"R² = {reginfo['R2']:.4f}")

        # Calcular Quc p/ recalque_critico
        quc_crit = calcular_quc(reg_coef, tipo_reg, recalque_critico)
        if idioma == 'Português':
            st.write(f"Quc para recalque crítico (0.1*D) = {quc_crit:.2f} tf")
        else:
            st.write(f"Quc for critical settlement (0.1*D) = {quc_crit:.2f} tf")

        # Se o usuário forneceu recalque_input > 0, calcula a carga
        if recalque_input > 0:
            quc_input = calcular_quc(reg_coef, tipo_reg, recalque_input)
            if idioma == 'Português':
                st.write(f"Carga para o recalque {recalque_input:.2f} mm = {quc_input:.2f} tf")
            else:
                st.write(f"Load for settlement {recalque_input:.2f} mm = {quc_input:.2f} tf")

        # Se o usuário forneceu carga_input > 0, calcula o recalque
        if carga_input > 0:
            # rig = rigidez_func(carga_input)
            rig = rigidez_func(carga_input)
            if rig > 0:
                rec = carga_input / rig
            else:
                rec = np.nan
            if idioma == 'Português':
                st.write(f"Recalque para a carga {carga_input:.2f} tf = {rec:.4f} mm")
            else:
                st.write(f"Settlement for load {carga_input:.2f} tf = {rec:.4f} mm")

    # Agora plotamos as interseções (o "X") e escrevemos no Streamlit
    # inters_adj[i] => interseção entre i e i+1
    for i, intersec in enumerate(inters_adj):
        if intersec is None:
            continue
        x_int, y_int = intersec
        # Plotar X
        plt.plot(x_int, y_int, 'mx', markersize=12, markeredgewidth=2)
        if idioma == 'Português':
            st.write(f"Interseção entre {num_romanos[i+1]} e {num_romanos[i+2]}: x={x_int:.3f}, rigidez={y_int:.3f}")
        else:
            st.write(f"Intersection between {num_romanos[i+1]} and {num_romanos[i+2]}: x={x_int:.3f}, stiffness={y_int:.3f}")

    # Finaliza matplotlib
    if idioma == 'Português':
        plt.xlabel('Carga (tf)')
        plt.ylabel('Rigidez (tf/mm)')
        plt.title('Regressões estendidas até interseções')
    else:
        plt.xlabel('Load (tf)')
        plt.ylabel('Stiffness (tf/mm)')
        plt.title('Extended regressions until intersections')

    plt.legend()
    st.pyplot(plt)


def calcular_regressao_estendida(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    1) Ajusta as regressões para cada subset escolhido.
    2) Coleta [tipo, coef, x_min, x_max, R² etc] em uma lista.
    3) Calcula interseções entre cada i e i+1 no overlap de [x_min_i, x_max_i] x [x_min_(i+1), x_max_(i+1)].
    4) Faz o plot de cada regressão do x_start ao x_end, onde x_start / x_end
       são definidos pelas interseções com i-1 / i+1 (se existirem).
    """
    # 1) Organiza e ajusta cada regressão
    info_regressoes = []
    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_reg = pontos_tipos[i]
        subset = tabela.iloc[lin_in : lin_fim+1].copy()

        # Domínio base
        x_min_i = subset['Carga'].min()
        x_max_i = subset['Carga'].max()

        if tipo_reg == 'linear':
            # polyfit em (Carga, rigidez)
            #   reg = [a, b] => y = a*x + b
            reg = np.polyfit(subset['Carga'], subset['rigidez'], deg=1)
            p = np.poly1d(reg)
            y_pred = p(subset['Carga'])
            y_obs = subset['rigidez']
            cor = np.corrcoef(y_pred, y_obs)[0,1]
            R2 = cor**2
        else:
            # 'log'
            #   reg = [a, b], com log(rigidez) = a*log10(Carga) + b
            reg = np.polyfit(subset['logQ'], subset['logRig'], deg=1)
            p = np.poly1d(reg)
            logRig_pred = p(subset['logQ'])
            y_pred = 10**(logRig_pred)
            y_obs = subset['rigidez']
            cor = np.corrcoef(y_pred, y_obs)[0,1]
            R2 = cor**2

        info_regressoes.append({
            'tipo': tipo_reg,
            'coef': reg,    # (a, b)
            'x_min': x_min_i,
            'x_max': x_max_i,
            'lin_in': lin_in,
            'lin_fim': lin_fim,
            'R2': R2
        })

    # 2) Calcular interseções entre i e i+1 (se num_regressoes > 1)
    inters_adj = [None]*(num_regressoes-1)  # inters_adj[i] = intersec entre i e i+1
    for i in range(num_regressoes-1):
        reg_i = info_regressoes[i]
        reg_j = info_regressoes[i+1]

        # Overlap de domínio
        xmin_overlap = max(reg_i['x_min'], reg_j['x_min'])
        xmax_overlap = min(reg_i['x_max'], reg_j['x_max'])
        if xmin_overlap < xmax_overlap:
            # Calcula interseções
            inters = calcular_interseccoes(
                reg_i['coef'], reg_j['coef'],
                reg_i['tipo'], reg_j['tipo'],
                xmin_overlap, xmax_overlap
            )
            if len(inters) > 0:
                # Pegamos a primeira (por exemplo)
                x_int, y_int = inters[0]
                inters_adj[i] = (x_int, y_int)
            else:
                inters_adj[i] = None
        else:
            inters_adj[i] = None

    # 3) Plotar regressões com extensões
    plot_regressoes(
        tabela,
        info_regressoes,
        inters_adj,
        idioma,
        carga_input,
        recalque_input,
        diametro_estaca
    )


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
        tabela['logQ'] = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq'] = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig'] = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        # Entradas de usuário
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

        tabela = tabela.sort_values(by="Carga").reset_index(drop=True)

        # Gráfico 1: Carga vs Recalque (Plotly)
        fig1 = px.scatter(
            tabela, x="Carga", y="Recalque",
            labels={
                "Carga": "Carga (tf)", 
                "Recalque": "Recalque (mm)"
            } if idioma == "Português" else {
                "Carga": "Load (tf)", 
                "Recalque": "Settlement (mm)"
            }
        )
        fig1.update_yaxes(autorange="reversed")
        fig1.update_layout(
            title="Carga vs Recalque" if idioma == "Português" else "Load vs Settlement",
            xaxis_title="Carga (tf)" if idioma == "Português" else "Load (tf)",
            yaxis_title="Recalque (mm)" if idioma == "Português" else "Settlement (mm)"
        )
        for i, row in tabela.iterrows():
            fig1.add_annotation(
                x=row["Carga"],
                y=row["Recalque"],
                text=str(i),
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-20
            )
        st.plotly_chart(fig1, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

        # Quantas regressões?
        num_regressoes = st.selectbox(
            'Quantas regressões:' if idioma == "Português" else 'How many regressions?', 
            [1, 2, 3, 4, 5], index=0
        )

        # Inputs para cada regressão (pontos inicial e final)
        pontos_tipos = []
        for i in range(num_regressoes):
            lin_in_key = f'lin_in_{i}'
            lin_fim_key = f'lin_fim_{i}'
            tipo_regressao_key = f'tipo_regressao_{i}'

            lin_in_default = st.session_state.get(lin_in_key, '0')
            lin_fim_default = st.session_state.get(lin_fim_key, str(len(tabela)-1))
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

            # Converter
            try:
                lin_in_val = int(lin_in_str)
                lin_fim_val = int(lin_fim_str)
            except:
                st.error("Entrada inválida: use números inteiros para os limites dos pontos.")
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

        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressao_estendida(
                tabela, num_regressoes, pontos_tipos,
                diametro_estaca, idioma,
                carga_input, recalque_input
            )

# Rode o programa
idioma = 'Português'  # ou 'English'
primeiro_programa(idioma)