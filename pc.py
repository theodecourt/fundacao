import streamlit as st
from curriculo import mostrar_curriculo  # Importa a função para exibir a foto
from programas import pagina_programas  # Importa a função da página Programas

# Função principal para a página
def pagina_principal():
    st.title('Luciano Decourt')
    
    st.write("Bem-vindo ao site de Luciano Decourt")

    # Carregar e exibir a imagem na página principal
    st.image("foto.jpeg", caption="Luciano Decourt", width=300)
    

# Função para a página "O engenheiro"
def pagina_engenheiro():
    st.title('Luciano Decourt')
    st.markdown(mostrar_curriculo(), unsafe_allow_html=True)

# Função para a página "Obras"
def pagina_obras():
    st.title('Obras de Luciano Decourt')
    st.write("Conteúdo de obras será adicionado aqui.")

# Função para a página "Artigos"
def pagina_artigos():
    st.title('Artigos de Luciano Decourt')
    st.write("Conteúdo de artigos será adicionado aqui.")

def main():
    st.sidebar.title("Navegação")

    # Exibe todas as opções como botões no menu lateral
    if st.sidebar.button("Página Principal"):
        pagina_principal()

    if st.sidebar.button("O engenheiro"):
        pagina_engenheiro()

    if st.sidebar.button("Obras"):
        pagina_obras()

    if st.sidebar.button("Artigos"):
        pagina_artigos()

    if st.sidebar.button("Programas"):
        pagina_programas()

if __name__ == '__main__':
    main()
