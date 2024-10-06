import streamlit as st
from primeiro_programa import primeiro_programa  # Importa o primeiro programa
from segundo_programa import segundo_programa  # Importa o segundo programa

# Função para a página "Programas"
def pagina_programas():
    st.title('Programas de Luciano Decourt')

    # Selectbox para selecionar qual programa será executado
    programa_selecionado = st.selectbox('Selecione o programa:', ['Interpretação de Provas de Carga', 'REC-0 (ZDM)'])

    # Chama o primeiro programa
    if programa_selecionado == 'Interpretação de Provas de Carga':
        primeiro_programa()

    # Chama o segundo programa
    elif programa_selecionado == 'REC-0 (ZDSM)':
        segundo_programa()
