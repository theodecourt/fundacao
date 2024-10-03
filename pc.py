import streamlit as st
from curriculo import mostrar_curriculo
from programas import pagina_programas
from videos import pagina_videos  # Importa a função para a página de vídeos

# Função principal para a página
def pagina_principal():
    st.title('Luciano Decourt')
    st.image("foto.jpeg", caption="Luciano Decourt", width=300)

def pagina_engenheiro():
    st.title('Luciano Decourt')
    st.markdown(mostrar_curriculo(), unsafe_allow_html=True)

def pagina_obras():
    st.title('Obras de Luciano Decourt')
    st.write("Conteúdo de obras será adicionado aqui.")

def pagina_artigos():
    st.title('Artigos de Luciano Decourt')
    st.write("Conteúdo de artigos será adicionado aqui.")

# Função principal para a navegação
def main():
    # Injetando CSS para alterar a cor do selectbox e o cursor
    st.markdown("""
        <style>
        select {
            cursor: pointer;
        }
        div[role="radiogroup"] > label > div:first-child {
            background-color: #003366;
        }
        div[role="radiogroup"] > label > div:first-child:hover {
            background-color: #002244;
        }
        div[role="radiogroup"] > label > div:first-child input:checked ~ div {
            background-color: #003366;
        }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("Navegação")
    st.sidebar.markdown("<h2 style='font-size: 24px;'>Selecione a página:</h2>", unsafe_allow_html=True)

    # Adicione o novo botão de vídeos no menu lateral
    pagina = st.sidebar.radio(
        "",
        ["Página Principal", "O Engenheiro", "Obras", "Artigos", "Programas", "Vídeos"]
    )

    # Exibe a página correspondente
    if pagina == "Página Principal":
        pagina_principal()
    elif pagina == "O Engenheiro":
        pagina_engenheiro()
    elif pagina == "Obras":
        pagina_obras()
    elif pagina == "Artigos":
        pagina_artigos()
    elif pagina == "Programas":
        pagina_programas()
    elif pagina == "Vídeos":
        pagina_videos()  # Chama a função da página de vídeos

if __name__ == '__main__':
    main()
