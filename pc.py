import streamlit as st
from curriculo import mostrar_curriculo  # Importa a função para exibir a foto
from programas import pagina_programas  # Importa a função da página Programas

# Função principal para a página
def pagina_principal():
    st.title('Luciano Decourt')
    
    # Carregar e exibir a imagem na página principal
    st.image("foto_luciano.jpg", caption="Luciano Decourt", width=300)
    
    st.write("Bem-vindo ao site de Luciano Decourt")

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

# Função principal para a navegação
def main():
    st.sidebar.title("Navegação")
    pagina = st.sidebar.selectbox(
        "Selecione a página:",
        ["Página Principal", "O engenheiro", "Obras", "Artigos", "Programas"]
    )

    if pagina == "Página Principal":
        pagina_principal()  # Chama a função da página principal
    elif pagina == "O engenheiro":
        pagina_engenheiro()
    elif pagina == "Obras":
        pagina_obras()
    elif pagina == "Artigos":
        pagina_artigos()
    elif pagina == "Programas":
        pagina_programas()  # Chama a função de programas importada de programas.py

if __name__ == '__main__':
    main()
