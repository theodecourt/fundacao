import streamlit as st
from primeiro_programa import primeiro_programa  # Importa o primeiro programa
from segundo_programa import segundo_programa  # Importa o segundo programa

# Função para a página "Programas"
def pagina_programas():
    st.title('Programas de Luciano Decourt')

    # Selectbox para selecionar qual programa será executado
    programa_selecionado = st.selectbox('Selecione o programa:', ['Programa 1', 'Programa 2'])

    # Chama o primeiro programa
    if programa_selecionado == 'Programa 1':
        primeiro_programa()

    # Chama o segundo programa
    elif programa_selecionado == 'Programa 2':
        segundo_programa()
