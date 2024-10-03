import streamlit as st
from curriculo import mostrar_curriculo
from programas import pagina_programas

# Função principal para a página
def pagina_principal():
    st.title('Luciano Decourt')
    st.write("Bem-vindo ao site de Luciano Decourt")
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

    # Exibe as páginas como botões no menu lateral
    if st.sidebar.button("Página Principal"):
        pagina_principal()
    elif st.sidebar.button("O engenheiro"):
        pagina_engenheiro()
    elif st.sidebar.button("Obras"):
        pagina_obras()
    elif st.sidebar.button("Artigos"):
        pagina_artigos()
    elif st.sidebar.button("Programas"):
        pagina_programas()

    # Exibe a página principal por padrão
    if "pagina" not in st.session_state:
        pagina_principal()

if __name__ == '__main__':
    main()
