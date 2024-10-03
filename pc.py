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

    # Adiciona um estilo CSS para manter todos os botões do mesmo tamanho
    st.sidebar.markdown(
        """
        <style>
        .button-style {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background-color: #008CBA;
            color: white;
            text-align: center;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            font-size: 16px;
            cursor: pointer;
        }
        .button-style:hover {
            background-color: #005f6b;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Cria links estilizados como botões
    if st.sidebar.markdown('<a href="#" class="button-style">Página Principal</a>', unsafe_allow_html=True):
        pagina_principal()

    if st.sidebar.markdown('<a href="#" class="button-style">O engenheiro</a>', unsafe_allow_html=True):
        pagina_engenheiro()

    if st.sidebar.markdown('<a href="#" class="button-style">Obras</a>', unsafe_allow_html=True):
        pagina_obras()

    if st.sidebar.markdown('<a href="#" class="button-style">Artigos</a>', unsafe_allow_html=True):
        pagina_artigos()

    if st.sidebar.markdown('<a href="#" class="button-style">Programas</a>', unsafe_allow_html=True):
        pagina_programas()

if __name__ == '__main__':
    main()
