import streamlit as st
from curriculo import mostrar_curriculo
from programas import pagina_programas

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
    st.sidebar.title("Navegação")

    st.sidebar.title("Selecione a página:")

    # Exibe as opções de navegação como Radio Button
    pagina = st.sidebar.radio(
        ["Página Principal", "O Engenheiro", "Obras", "Artigos", "Programas"]
    )

    # Exibe a página correspondente
    if pagina == "Página Principal":
        pagina_principal()
    elif pagina == "O engenheiro":
        pagina_engenheiro()
    elif pagina == "Obras":
        pagina_obras()
    elif pagina == "Artigos":
        pagina_artigos()
    elif pagina == "Programas":
        pagina_programas()

if __name__ == '__main__':
    main()
