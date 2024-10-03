import streamlit as st
from curriculo import mostrar_curriculo
from programas import pagina_programas
from videos import pagina_videos

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
        /* Mudar cursor do selectbox para pointer (mãozinha) */
        .stSelectbox > div > div:first-child {
            cursor: pointer !important;
        }
        div[role="radiogroup"] > label > div:first-child {
            background-color: #003366;  /* Cor azul escuro */
        }
        div[role="radiogroup"] > label > div:first-child:hover {
            background-color: #002244;  /* Azul escuro mais forte ao passar o mouse */
        }
        div[role="radiogroup"] > label > div:first-child input:checked ~ div {
            background-color: #003366;  /* Cor azul escuro quando selecionado */
        }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("Navegação")

    # Exibe o título de seleção com tamanho maior usando HTML
    st.sidebar.markdown("<h2 style='font-size: 24px;'>Selecione a página:</h2>", unsafe_allow_html=True)

    # Exibe as opções de navegação como Radio Button
    pagina = st.sidebar.radio(
        "",  # O label vazio, pois já colocamos o título personalizado acima
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
