import streamlit as st
from primeiro_programa import primeiro_programa  # Importa o primeiro programa
from segundo_programa import segundo_programa  # Importa o segundo programa

# Função para a página "Programas"
def pagina_programas(idioma):
    if idioma == "Português":
        st.title('Programas de Luciano Decourt')

        # Selectbox para selecionar qual programa será executado
        programa_selecionado = st.selectbox('Selecione o programa:', ['Interpretação de Provas de Carga', 'REC-0'])

        # Chama o primeiro programa
        if programa_selecionado == 'Interpretação de Provas de Carga':
            primeiro_programa(idioma)

        # Chama o segundo programa
        elif programa_selecionado == 'REC-zero':
            segundo_programa(idioma)
    else:
        st.title('Programs of Luciano Décourt')

        # Selectbox to select which program will be executed
        program_selected = st.selectbox('Select the program:', ['Load Test Interpretation', 'ZDSM'])

        # Calls the first program
        if program_selected == 'Load Test Interpretation':
            primeiro_programa(idioma)

        # Calls the second program
        elif program_selected == 'ZDSM':
            segundo_programa(idioma)

