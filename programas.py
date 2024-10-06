import streamlit as st
from primeiro_programa import primeiro_programa
from segundo_programa import segundo_programa

# Função para a página "Programas"
def pagina_programas():
    st.title('Programas de Luciano Decourt')

    programa_selecionado = st.selectbox('Selecione o programa:', ['Programa 1', 'Programa 2'])

    if programa_selecionado == 'Programa 1':
        primeiro_programa()  # Chama o primeiro programa importado de primeiro_programa.py
    elif programa_selecionado == 'Programa 2':
        segundo_programa()  # Chama o segundo programa importado de segundo_programa.py
