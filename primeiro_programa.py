import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import math
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy.optimize import fsolve

# Função para carregar a tabela
def carregar_tabela():
    arq = input('Qual é o nome do arquivo? ')
    arq = arq + ".csv"
    return pd.read_csv(arq, delimiter=';')

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

# Função para calcular a regressão e plotar os gráficos
def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca):
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    
    colors = ['b', 'red', 'green']
    plt.plot(x0, y0, 'go', label='Dados Originais')
    
    regressions = []
    tipos = []
    
    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_regressao = pontos_tipos[i]
        linear = tabela[lin_in-1:lin_fim]  # Ajuste para começar do ponto 1
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

        plt.plot(x, y, colors[i], label=f'Regressão {i+1}')
        
        regressions.append(reg)
        tipos.append(tipo_regressao)
        
        print(f'Pontos utilizados na regressão {i+1}: ', lin_in, ' até ', lin_fim)
        print('Tipo de regressão: ', tipo_regressao.capitalize())
        print('Equação da regressão: ', equacao)
        print('R²: ', R_sq)
    
    # Calcular e mostrar pontos de interseção entre todas as combinações possíveis
    for i in range(num_regressoes):
        for j in range(i + 1, num_regressoes):
            interseccao = calcular_interseccao(regressions[i], regressions[j], tipos[i], tipos[j])
            plt.plot(interseccao[0], interseccao[1], 'rx')  # Marca a interseção com um 'x' vermelho
            print(f'Interseção entre regressão {i+1} e {j+1}: Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}')
    
    # Calcular Quc
    recalque_critico = 0.1 * diametro_estaca
    for i, reg in enumerate(regressions):
        if tipos[i] == 'linear':
            a = reg[1]
            b = reg[0]
            quc = a / ((1 / recalque_critico) - b)
        else:  # log
            def func_quc_log(x):
                return 10**(reg[0] * np.log10(x) + reg[1]) - (x / recalque_critico)
            quc = fsolve(func_quc_log, x0=1)[0]
        print(f'Quc para a regressão {i+1}: {quc:.4f}')
    
    plt.xlabel('Carga')
    plt.ylabel('Rigidez')
    plt.title('Regressão de Carga x Rigidez')
    plt.legend().set_visible(False)  # Oculta a caixa de legenda
    plt.show()

# Função para criar widgets de entrada para os pontos iniciais e finais e tipo de regressão
def criar_widgets_pontos_tipos(num_regressoes):
    widgets_pontos_tipos = []
    for i in range(num_regressoes):
        lin_in = widgets.IntText(description=f'Ponto inicial {i+1}:', value=1)  # Ajuste para iniciar do ponto 1
        lin_fim = widgets.IntText(description=f'Ponto final {i+1}:')
        tipo_regressao = widgets.Dropdown(
            options=['linear', 'log'],
            value='linear',
            description=f'Tipo {i+1}:',
        )
        widgets_pontos_tipos.append((lin_in, lin_fim, tipo_regressao))
    return widgets_pontos_tipos

# Função para exibir widgets de entrada para os pontos iniciais e finais e tipo de regressão
def exibir_widgets_pontos_tipos(widgets_pontos_tipos):
    for lin_in, lin_fim, tipo_regressao in widgets_pontos_tipos:
        display(lin_in)
        display(lin_fim)
        display(tipo_regressao)

# Função para obter os valores dos widgets de entrada
def obter_valores_widgets(widgets_pontos_tipos):
    pontos_tipos = []
    for lin_in, lin_fim, tipo_regressao in widgets_pontos_tipos:
        pontos_tipos.append((lin_in.value, lin_fim.value, tipo_regressao.value))
    return pontos_tipos

# Função principal para executar o fluxo do primeiro programa
def primeiro_programa():
    tabela = carregar_tabela()
    print(tabela.columns)
    
    # Pergunta o diâmetro da estaca
    diametro_estaca = float(input('Qual é o diâmetro da estaca? '))

    # Plota os gráficos antes de exibir as opções de regressões
    fig = px.scatter(tabela, x="Carga", y="Recalque")
    fig.update_yaxes(autorange="reversed")
    fig.show()

    tabela['rigidez'] = tabela.apply(lambda row: row.Carga / row.Recalque, axis=1)
    fig2 = px.scatter(tabela, x="Carga", y="rigidez")
    fig2.show()

    tabela['logQ'] = tabela.apply(lambda row: math.log(row.Carga, 10), axis=1)
    tabela['logReq'] = tabela.apply(lambda row: math.log(row.Recalque, 10), axis=1)
    tabela['logRig'] = tabela.apply(lambda row: math.log(row.rigidez, 10), axis=1)
    
    # Dropdown para selecionar o número de regressões
    dropdown = widgets.Dropdown(
        options=[1, 2, 3],
        value=1,
        description='Quantas regressões:',
    )
    display(dropdown)
    
    # Caixas de entrada para os pontos iniciais e finais e tipo de regressão
    widgets_pontos_tipos = criar_widgets_pontos_tipos(1)
    exibir_widgets_pontos_tipos(widgets_pontos_tipos)
    
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            num_regressoes = change['new']
            clear_output(wait=True)
            display(dropdown)
            
            # Exibe as caixas de entrada conforme o número de regressões selecionadas
            widgets_pontos_tipos = criar_widgets_pontos_tipos(num_regressoes)
            exibir_widgets_pontos_tipos(widgets_pontos_tipos)
            
            button = widgets.Button(description="Calcular Regressões")
            display(button)
            
            def on_button_click(b):
                pontos_tipos = obter_valores_widgets(widgets_pontos_tipos)
                calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca)
            
            button.on_click(on_button_click)
    
    dropdown.observe(on_change)
    
    button = widgets.Button(description="Calcular Regressões")
    display(button)
    
    def on_button_click(b):
        pontos_tipos = obter_valores_widgets(widgets_pontos_tipos)
        calcular_regressao(tabela, dropdown.value, pontos_tipos, diametro_estaca)
    
    button.on_click(on_button_click)
