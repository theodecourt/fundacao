import streamlit as st
from primeiro_programa import primeiro_programa  # Importa o primeiro programa
from segundo_programa import segundo_programa  # Importa o segundo programa

# Função para injetar CSS personalizado
def set_background_and_text_color():
    st.markdown(
        """
        <style>
        /* Define o fundo da página */
        body {
            background-color: #000000; /* Preto */
            color: #FFFFFF; /* Branco */
        }

        /* Define o fundo da barra lateral */
        .css-1d391kg { /* Pode variar dependendo da versão do Streamlit */
            background-color: #000000;
        }

        /* Define a cor do texto nos elementos do Streamlit */
        .css-1v0mbdj {
            color: #FFFFFF;
        }

        /* Ajusta outros elementos conforme necessário */
        </style>
        """,
        unsafe_allow_html=True
    )

# Chama a função para aplicar o CSS
set_background_and_text_color()

# Função para a página "Programas"
def pagina_programas(idioma):
    if idioma == "Português":
        st.title('Programas de Luciano Décourt')

        # Selectbox para selecionar qual programa será executado
        programa_selecionado = st.selectbox('Selecione o programa:', ['Interpretação de Provas de Carga', 'REC-zero'])

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

# Exemplo de como chamar a função pagina_programas
# Defina o idioma conforme necessário, por exemplo:
idioma_selecionado = st.sidebar.selectbox('Selecione o idioma:', ['Português', 'English'])
pagina_programas(idioma_selecionado)
