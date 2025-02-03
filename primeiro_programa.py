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
            st.error(f"Erro ao carregar o arquivo: {e}" if idioma == "Português" else f"Error loading file: {e}")
            return None
    botao_download_exemplo(idioma)
    return None

def calcular_interseccao(reg1, reg2, tipo1, tipo2, x_min, x_max):
    """
    Tenta calcular TODAS as interseções entre duas regressões (linear ou log),
    se existirem, dentro do intervalo [x_min, x_max].
    Retorna uma lista de (x_int, y_int).
    """
    interseccoes = []
    
    def esta_no_intervalo(x):
        return (x_min <= x <= x_max)
    
    def esta_no_intervalo_positivo(x):
        return (x > 0) and (x_min <= x <= x_max)

    # =============== CASO 1: Ambas lineares ===============
    if tipo1 == 'linear' and tipo2 == 'linear':
        # reg = [a, b] no polyfit => y = a*x + b
        # Mas atenção! Se você usou np.polyfit(subset['Carga'], subset['rigidez'], deg=1),
        # então reg[0] = coef_angular, reg[1] = coef_linear
        a1, b1 = reg1  # rigidez = a1*x + b1
        a2, b2 = reg2

        if np.isclose(a1, a2, atol=1e-14):
            return []  # paralelas (sem ponto único ou são coincidentes)

        # a1*x + b1 = a2*x + b2 => x*(a1 - a2) = (b2 - b1)
        x_int = (b2 - b1) / (a1 - a2)
        y_int = a1*x_int + b1

        if esta_no_intervalo(x_int):
            interseccoes.append([x_int, y_int])

    # =============== CASO 2: Ambas logarítmicas ===============
    elif tipo1 == 'log' and tipo2 == 'log':
        # reg = [a, b] => log(rig) = a * log(Q) + b
        # => rig = 10^(a*log10(Q) + b)
        a1, b1 = reg1
        a2, b2 = reg2

        if np.isclose(a1, a2, atol=1e-14):
            return []

        # a1*log10(x) + b1 = a2*log10(x) + b2
        # (a1 - a2)*log10(x) = b2 - b1
        # log10(x) = (b2 - b1)/(a1 - a2)
        log_x = (b2 - b1) / (a1 - a2)
        x_int = 10**(log_x)
        y_int = 10**(a1*log_x + b1)

        if esta_no_intervalo_positivo(x_int):
            interseccoes.append([x_int, y_int])

    # =============== CASO 3: Linear + Log ===============
    else:
        # Identificar quem é linear e quem é log
        if tipo1 == 'linear':
            a_lin, b_lin = reg1  # rig = a_lin*x + b_lin
            a_log, b_log = reg2  # rig = 10^(a_log*log10(x) + b_log)
        else:
            a_lin, b_lin = reg2
            a_log, b_log = reg1
        
        def func_intersec(x):
            if x <= 0:
                return np.nan
            val_lin = a_lin*x + b_lin
            val_log = 10**(a_log * np.log10(x) + b_log)
            return val_lin - val_log
        
        # Usaremos uma busca de raiz segmentada no intervalo [x_min, x_max].
        num_steps = 200
        xs = np.linspace(x_min, x_max, num_steps)
        fs = []
        for xi in xs:
            fs.append(func_intersec(xi))

        for i in range(len(xs)-1):
            f1, f2 = fs[i], fs[i+1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 * f2 < 0:  # mudança de sinal => raiz no meio
                try:
                    raiz = brentq(func_intersec, xs[i], xs[i+1], xtol=1e-12)
                    if esta_no_intervalo_positivo(raiz):
                        # y_lin ou y_log, devem ser iguais no ponto
                        y_raiz = a_lin*raiz + b_lin
                        interseccoes.append([raiz, y_raiz])
                except ValueError:
                    pass

    return interseccoes

def calcular_quc(reg, tipo_regressao, valor_critico):
    """
    Cálculo de Quc (carga para um determinado recalque) dependendo do tipo de regressão.
    *** Verifique se faz sentido para seu caso. ***
    """
    if tipo_regressao == 'linear':
        # Se a regressão é: rigidez = a*x + b
        # rigidez = Carga/Recalque => (Carga/Recalque) = a*Carga + b
        # => ??? (depende de qual fórmula você quer usar)
        # Abaixo está um exemplo simplificado que talvez não faça sentido físico,
        # então adapte ao seu caso. Mantive seu código, mas tome cuidado:

        a = reg[0]  # coef_angular
        b = reg[1]  # coef_linear
        # rigidez = a*C + b = C / recalque
        # => C / recalque = a*C + b
        # => C (1/(recalque)) = a*C + b
        # => C*(1/recalque - a) = b
        # => C = b / (1/recalque - a)
        # Vamos implementar isso:
        try:
            quc = b / ((1/valor_critico) - a)
        except ZeroDivisionError:
            quc = np.nan

    else:  # log
        # rigidez = 10^(a*log10(C) + b)
        # => C / recalque = 10^(a*log10(C) + b)
        # => G(C) = 10^(...) - C/recalque
        # Precisamos resolver G(C) = 0
        def func_quc_log(x):
            if x <= 0:
                return np.nan
            rig_log = 10**(reg[0] * np.log10(x) + reg[1])
            return rig_log - (x / valor_critico)
        try:
            # Ajustar limites de busca conforme seu caso:
            quc = brentq(func_quc_log, 1e-6, 1e9, xtol=1e-8)
        except ValueError:
            quc = np.nan
    return quc

def calcular_regressoes_e_plotar(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma, carga_input, recalque_input):
    """
    Faz em duas etapas:
      1) Ajusta as regressões (guardando parâmetros e intervalos de dados).
      2) Calcula as interseções entre regressões consecutivas.
      3) Plota cada regressão do limite inferior (interseção ou minPlotX) até
         o limite superior (interseção ou maxPlotX).

    Ao final, gera o gráfico usando matplotlib.
    """

    # Ordenar a tabela pela Carga
    tabela = tabela.sort_values(by='Carga').reset_index(drop=True)

    # Intervalo global de plotagem (um pouco além dos dados, para "sair da tela")
    min_x_data = tabela['Carga'].min()
    max_x_data = tabela['Carga'].max()
    margem = 0.2 * (max_x_data - min_x_data) if (max_x_data > min_x_data) else 1.0
    min_plot_x = max(min_x_data - margem, 1e-10)  # se quiser evitar log(0)
    max_plot_x = max_x_data + margem

    # Lista para guardar os resultados das regressões
    # Cada item: {
    #   'reg': (a, b),
    #   'tipo': 'linear' ou 'log',
    #   'subset_xmin': menor Carga do subset,
    #   'subset_xmax': maior Carga do subset
    # }
    lista_regs = []

    # Ajustar cada regressão
    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_reg = pontos_tipos[i]
        subset = tabela.iloc[lin_in:lin_fim+1]

        if tipo_reg == 'linear':
            # polyfit em x = Carga, y = rigidez
            reg = np.polyfit(subset['Carga'], subset['rigidez'], deg=1)  # (a, b)
            # reg[0] = a, reg[1] = b => rig = a*x + b
        else:
            # log
            reg = np.polyfit(subset['logQ'], subset['logRig'], deg=1)
            # reg[0] = a, reg[1] = b => log10(rig) = a*log10(Q) + b

        x_min_sub = subset['Carga'].min()
        x_max_sub = subset['Carga'].max()

        lista_regs.append({
            'reg': reg,
            'tipo': tipo_reg,
            'subset_xmin': x_min_sub,
            'subset_xmax': x_max_sub
        })

    # Agora, calcular as interseções entre regressões consecutivas
    # inters[i] será a interseção entre regressão i e i+1 (se existir)
    inters = [None] * (num_regressoes - 1)

    for i in range(num_regressoes - 1):
        r1 = lista_regs[i]['reg']
        t1 = lista_regs[i]['tipo']
        r2 = lista_regs[i+1]['reg']
        t2 = lista_regs[i+1]['tipo']

        # Precisamos do overlap de x para buscar a interseção
        x_min_1 = lista_regs[i]['subset_xmin']
        x_max_1 = lista_regs[i]['subset_xmax']
        x_min_2 = lista_regs[i+1]['subset_xmin']
        x_max_2 = lista_regs[i+1]['subset_xmax']

        # Overlap do subset
        overlap_min = max(x_min_1, x_min_2, min_plot_x)
        overlap_max = min(x_max_1, x_max_2, max_plot_x)

        if overlap_min < overlap_max:
            # Encontrar interseccoes no overlap
            pts_inters = calcular_interseccao(r1, r2, t1, t2, overlap_min, overlap_max)
            if len(pts_inters) > 0:
                # Se houver mais de uma interseção, pegamos a primeira
                # ou poderíamos escolher a que estiver dentro do overlap.
                inters[i] = pts_inters[0]
            else:
                inters[i] = None
        else:
            inters[i] = None

    # Iniciar o plot
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 6))

    # Plotar os pontos originais
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    plt.plot(x0, y0, 'go', label='Dados Originais' if idioma == 'Português' else 'Original Data')

    # Numerar os pontos no gráfico
    for i, (x, y) in enumerate(zip(x0, y0)):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center')

    recalque_critico = 0.1 * diametro_estaca  # se for este o conceito de Quc

    # Plotar cada regressão
    for i in range(num_regressoes):
        reg_info = lista_regs[i]
        reg = reg_info['reg']
        tipo_reg = reg_info['tipo']

        # Limite inferior de plot
        if i == 0:
            x_start = min_plot_x
        else:
            if inters[i-1] is not None:
                x_start = inters[i-1][0]
            else:
                x_start = min_plot_x

        # Limite superior de plot
        if i == num_regressoes - 1:
            x_end = max_plot_x
        else:
            if inters[i] is not None:
                x_end = inters[i][0]
            else:
                x_end = max_plot_x

        # Só faz sentido se x_start < x_end
        if x_start < x_end:
            x_vals = np.linspace(x_start, x_end, 300)
            if tipo_reg == 'linear':
                a, b = reg  # rig = a*C + b
                y_vals = a*x_vals + b
            else:
                a, b = reg
                # rig = 10^(a*log10(x) + b)
                # cuidado com x <= 0, mas assumimos x>0
                y_vals = 10**(a*np.log10(x_vals) + b)

            plt.plot(x_vals, y_vals, color=colors[i], label=f'Regressão {num_romanos[i+1]}' if idioma == 'Português' else f'Regression {i+1}')

            # Rótulo (número romano) no "meio" do intervalo [x_start, x_end]
            x_centro = (x_start + x_end) / 2
            if tipo_reg == 'linear':
                y_centro = a*x_centro + b
            else:
                y_centro = 10**(a*np.log10(x_centro) + b)

            # Desenhar o texto
            plt.text(
                x_centro,
                y_centro * 1.05,  # pequeno deslocamento
                f'{num_romanos[i+1]}',
                color=colors[i],
                fontsize=20,
                fontweight='bold',
                ha='center'
            )

        # Cálculo do R² para exibir
        # Precisamos usar os pontos do subset usado para este ajuste
        # (não do x_vals "artificial").
        lin_in, lin_fim, _ = pontos_tipos[i]
        subset_local = tabela.iloc[lin_in:lin_fim+1]
        if tipo_reg == 'linear':
            pred = reg[0]*subset_local['Carga'] + reg[1]
        else:
            # log
            a, b = reg
            pred = 10**(a*subset_local['logQ'] + b)

        obs = subset_local['rigidez']
        corr_matrix = np.corrcoef(pred, obs)
        corr = corr_matrix[0, 1]
        R_sq = corr**2

        # Exibir infos
        cor_legenda = colors[i]
        if idioma == "Português":
            st.markdown(
                f"<b style='color:{cor_legenda};'>Pontos na regressão {num_romanos[i+1]}: {lin_in} até {lin_fim}</b>",
                unsafe_allow_html=True
            )
            st.write('Tipo de regressão:', tipo_reg.capitalize())
            if tipo_reg == 'linear':
                st.write(f'Equação: rig = {reg[0]:.4f} * C + {reg[1]:.4f}')
            else:
                st.write(f'Equação: log(rig) = {reg[0]:.4f} * log(C) + {reg[1]:.4f}')
            st.write('R²:', R_sq)
            quc = calcular_quc(reg, tipo_reg, recalque_critico)
            st.write(f'Quc (para {0.1}*D = {recalque_critico:.2f} mm): {quc:.2f} tf')
        else:
            st.markdown(
                f"<b style='color:{cor_legenda};'>Points in regression {num_romanos[i+1]}: {lin_in} to {lin_fim}</b>",
                unsafe_allow_html=True
            )
            st.write('Regression type:', tipo_reg.capitalize())
            if tipo_reg == 'linear':
                st.write(f'Equation: rig = {reg[0]:.4f} * Load + {reg[1]:.4f}')
            else:
                st.write(f'Equation: log(rig) = {reg[0]:.4f} * log(Load) + {reg[1]:.4f}')
            st.write('R²:', R_sq)
            quc = calcular_quc(reg, tipo_reg, recalque_critico)
            st.write(f'Quc (for 0.1*D = {recalque_critico:.2f} mm): {quc:.2f} tf')

        # Se o usuário forneceu recalque_input > 0, calcula a carga para esse recalque
        if recalque_input > 0:
            carga_calculada = calcular_quc(reg, tipo_reg, recalque_input)
            if idioma == "Português":
                st.write(f"A carga para o recalque {recalque_input:.2f} mm (usando {num_romanos[i+1]}) é {carga_calculada:.2f} tf.")
            else:
                st.write(f"The load for settlement {recalque_input:.2f} mm (using {num_romanos[i+1]}) is {carga_calculada:.2f} tf.")

        # Se o usuário forneceu carga_input > 0, calcula o recalque
        if carga_input > 0:
            if tipo_reg == 'linear':
                rigidez_calc = reg[0]*carga_input + reg[1]
            else:
                rigidez_calc = 10**(reg[0]*np.log10(carga_input) + reg[1])
            recalque_calculado = carga_input / rigidez_calc
            if idioma == "Português":
                st.write(f"Para a carga de {carga_input:.2f} tf (usando {num_romanos[i+1]}), o recalque será {recalque_calculado:.2f} mm.")
            else:
                st.write(f"For load {carga_input:.2f} tf (using {num_romanos[i+1]}), settlement will be {recalque_calculado:.2f} mm.")

    # Plotar as interseções
    for i, val in enumerate(inters):
        if val is not None:
            x_int, y_int = val
            plt.plot(
                x_int, 
                y_int,
                marker='x',
                markersize=12,
                markeredgewidth=3,
                color='magenta'
            )
            if idioma == "Português":
                plt.text(x_int, y_int, f"({x_int:.2f}, {y_int:.2f})", color='magenta', fontsize=10, ha='left')
                st.write(f"Interseção entre {num_romanos[i+1]} e {num_romanos[i+2]}: Carga={x_int:.2f}, Rigidez={y_int:.2f}")
            else:
                plt.text(x_int, y_int, f"({x_int:.2f}, {y_int:.2f})", color='magenta', fontsize=10, ha='left')
                st.write(f"Intersection between {num_romanos[i+1]} and {num_romanos[i+2]}: Load={x_int:.2f}, Stiffness={y_int:.2f}")

    # Finalizar plot
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
    # Carrega a tabela
    tabela = carregar_tabela(idioma)
    if tabela is not None:
        # Renomear colunas se necessário
        if "Carga (tf)" in tabela.columns and "Recalque (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Carga (tf)": "Carga", "Recalque (mm)": "Recalque"})
        elif "Load (tf)" in tabela.columns and "Settlement (mm)" in tabela.columns:
            tabela = tabela.rename(columns={"Load (tf)": "Carga", "Settlement (mm)": "Recalque"})
        else:
            msg_erro = (
                "Formato de coluna inválido. Certifique-se de que o arquivo contém 'Carga (tf)' e 'Recalque (mm)' "
                "ou 'Load (tf)' e 'Settlement (mm)'."
            )
            st.error(msg_erro if idioma == "Português" else "Invalid column format. Make sure the file has 'Load (tf)' and 'Settlement (mm)' columns.")
            return
        
        # Converter para numérico
        try:
            tabela['Carga'] = tabela['Carga'].astype(float)
            tabela['Recalque'] = tabela['Recalque'].astype(float)
        except ValueError:
            st.error("As colunas 'Carga' e 'Recalque' devem conter apenas valores numéricos." if idioma=="Português" 
                     else "Columns 'Carga' and 'Recalque' must contain only numeric values.")
            return

        # Calcular rigidez e logs
        tabela['rigidez'] = tabela['Carga'] / tabela['Recalque']
        tabela['logQ'] = tabela['Carga'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logReq'] = tabela['Recalque'].apply(lambda x: math.log10(x) if x > 0 else np.nan)
        tabela['logRig'] = tabela['rigidez'].apply(lambda x: math.log10(x) if x > 0 else np.nan)

        # Inputs
        diametro_estaca = st.number_input(
            'Qual é o diâmetro da estaca? (mm)' if idioma == "Português" else 'What is the pile diameter? (mm)', 
            min_value=0.01, format="%.2f"
        )

        recalque_input = st.number_input(
            'Quer calcular a carga para qual recalque? (mm)' if idioma == "Português" else 'Which settlement do you want to calculate the load for? (mm)',
            format="%.2f", min_value=0.0
        )
        carga_input = st.number_input(
            'Quer estimar o recalque para qual carga? (tf)' if idioma == "Português" else 'Which load do you want to estimate settlement for? (tf)',
            format="%.2f", min_value=0.0
        )

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

        # Lê os intervalos e tipo de regressão de cada uma
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
                st.error(f"Entrada inválida para o ponto inicial da regressão {num_romanos[i+1]}. Insira um número inteiro." if idioma=="Português"
                         else f"Invalid input for the start point of regression {num_romanos[i+1]}. Enter an integer.")
                return
            try:
                lin_fim_val = int(lin_fim_str)
            except ValueError:
                st.error(f"Entrada inválida para o ponto final da regressão {num_romanos[i+1]}. Insira um número inteiro." if idioma=="Português"
                         else f"Invalid input for the end point of regression {num_romanos[i+1]}. Enter an integer.")
                return

            if lin_in_val < 0 or lin_in_val >= len(tabela):
                st.error(f"Ponto inicial {lin_in_val} fora dos limites (0 a {len(tabela)-1})." if idioma=="Português"
                         else f"Start point {lin_in_val} out of bounds (0 to {len(tabela)-1}).")
                return
            if lin_fim_val < lin_in_val or lin_fim_val >= len(tabela):
                st.error(f"Ponto final {lin_fim_val} deve estar entre {lin_in_val} e {len(tabela)-1}." if idioma=="Português"
                         else f"End point {lin_fim_val} must be between {lin_in_val} and {len(tabela)-1}.")
                return

            tipo_reg_val = st.selectbox(
                f'Tipo de regressão {num_romanos[i+1]}:' if idioma == "Português" else f'Regression type {num_romanos[i+1]}:', 
                ['linear', 'log'], 
                index=0,
                key=tipo_regressao_key
            )

            pontos_tipos.append((lin_in_val, lin_fim_val, tipo_reg_val))

        # Botão para calcular
        if st.button('Calcular Regressões' if idioma == "Português" else 'Calculate Regressions'):
            calcular_regressoes_e_plotar(
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