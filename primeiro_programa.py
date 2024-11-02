# Função para calcular e plotar as regressões ajustadas
def calcular_regressao(tabela, num_regressoes, pontos_tipos, diametro_estaca, idioma):
    x0 = tabela['Carga']
    y0 = tabela['rigidez']
    
    colors = ['b', 'red', 'green']
    plt.plot(x0, y0, 'go', label='Dados Originais')
    
    regressions = []
    tipos = []
    
    recalque_critico = 0.1 * diametro_estaca  # Cálculo do recalque crítico

    for i in range(num_regressoes):
        lin_in, lin_fim, tipo_regressao = pontos_tipos[i]
        linear = tabela[lin_in-1:lin_fim]
        if tipo_regressao == 'linear':
            reg = np.polyfit(linear['Carga'], linear['rigidez'], deg=1)
            predict = np.poly1d(reg)
            corr_matrix = np.corrcoef(linear['rigidez'], linear['Carga'])

            if i == 0 and num_regressoes > 1:
                # Calcula a interseção entre a primeira e a segunda regressão
                reg2, tipo_reg2 = regressions[1], tipos[1]
                interseccao = calcular_interseccao(reg, reg2, tipo_regressao, tipo_reg2)
                x = np.linspace(tabela['Carga'].iloc[lin_in-1], interseccao[0], 100)
            else:
                x = np.linspace(tabela['Carga'].iloc[lin_in-1], tabela['Carga'].max(), 100)

            y = predict(x)

            if idioma == "Português":
                equacao = f'rigidez (tf/mm) = {reg[0]:.4f} * Carga (tf) + {reg[1]:.4f}'
            else:
                equacao = f'stiffness (tf/mm) = {reg[0]:.4f} * Load (tf) + {reg[1]:.4f}'

        else:  # log
            reg = np.polyfit(linear['logQ'], linear['logRig'], deg=1)
            predict = np.poly1d(reg)
            x = np.linspace(0.1, tabela['Carga'].max(), 100)  # Evitar log(0)
            y = 10**predict(np.log10(x))
            corr_matrix = np.corrcoef(linear['logRig'], linear['logQ'])
            equacao = f'log(rigidez) = {reg[0]:.4f} * log(Carga) + {reg[1]:.4f}'
        
        corr = corr_matrix[0, 1]
        R_sq = corr**2

        # Calcular o Quc
        quc = calcular_quc(reg, tipo_regressao, recalque_critico)

        plt.plot(x, y, colors[i], label=f'Regressão {i+1}')
        
        if idioma == "Português":
            st.write(f'Pontos utilizados na regressão {num_romanos[i+1]}: {lin_in} até {lin_fim}')
            st.write('Tipo de regressão:', tipo_regressao.capitalize())
            st.write('Equação da regressão:', equacao)
            st.write('R²:', R_sq)
            st.write(f'Quc para a regressão {num_romanos[i+1]}: {quc:.2f} tf')
        else:
            st.write(f'Points used in regression {num_romanos[i+1]}: {lin_in} to {lin_fim}')
            st.write('Regression type:', tipo_regressao.capitalize())
            st.write('Regression equation:', equacao)
            st.write('R²:', R_sq)
            st.write(f'Quc for regression {num_romanos[i+1]}: {quc:.2f} tf')

        # Adiciona a regressão e o tipo de regressão à lista
        regressions.append(reg)
        tipos.append(tipo_regressao)

    # Calcula a interseção se houver pelo menos duas regressões
    if num_regressoes >= 2:
        interseccao = calcular_interseccao(regressions[0], regressions[1], tipos[0], tipos[1])
        plt.plot(interseccao[0], interseccao[1], 'rx')  # Marca a interseção com um 'x' vermelho
        if idioma == "Português":
            st.write(f'Interseção entre regressão 1 e 2: Carga = {interseccao[0]:.4f}, Rigidez = {interseccao[1]:.4f}')
        else:
            st.write(f'Intersection between regression 1 and 2: Load = {interseccao[0]:.4f}, Stiffness = {interseccao[1]:.4f}')
    
    if idioma == "Português":
        plt.xlabel('Carga')
        plt.ylabel('Rigidez')
        plt.title('Regressão de Carga x Rigidez')
    else:
        plt.xlabel('Load')
        plt.ylabel('Stiffness')
        plt.title('Load vs Stiffness Regression')

    plt.legend()
    st.pyplot(plt)

