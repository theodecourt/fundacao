import streamlit as st
from curriculo import mostrar_curriculo
from programas import pagina_programas
from artigos import pagina_artigos
from videos import pagina_videos
from citacoes import pagina_citacoes

# Dicionários de tradução
textos = {
    "pt": {
        "selecione_pagina": "Selecione a página:",
        "pagina_principal": "Página Principal",
        "engenheiro": "O Engenheiro",
        "obras": "Obras",
        "artigos": "Artigos",
        "programas": "Programas",
        "videos": "Vídeos",
        "titulo_principal": "Luciano Décourt",
        "bem_vindo": "Bem-vindo ao site de Luciano Décourt",
        "citações" : "Citações"
    },
    "en": {
        "selecione_pagina": "Select the page:",
        "pagina_principal": "Home",
        "engenheiro": "The Engineer",
        "obras": "Works",
        "artigos": "Articles",
        "programas": "Programs",
        "videos": "Videos",
        "titulo_principal": "Luciano Décourt",
        "bem_vindo": "Welcome to Luciano Décourt's website",
        "citações" : "Citations"
    }
}

# Função principal para a página
def pagina_principal(texto):
    st.title(texto["titulo_principal"])
    st.image("foto.jpeg", caption="Luciano Décourt", width=300)
    st.write(texto["bem_vindo"])

def pagina_engenheiro(texto):
    st.title(texto["titulo_principal"])
    st.markdown(mostrar_curriculo(), unsafe_allow_html=True)

def pagina_obras(texto):
    st.title(texto["obras"])
    st.write("Conteúdo de obras será adicionado aqui.")

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

        /* Aumentar o tamanho do texto do seletor de idioma */
        label[for="Choose Language"] {
            font-size: 24px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    # Escolha de idioma
    idioma = st.sidebar.selectbox("Choose Language", ["Português", "English"])
    
    # Define o dicionário de textos conforme o idioma
    if idioma == "Português":
        texto = textos["pt"]
    else:
        texto = textos["en"]

    st.sidebar.title("Navegação")

    # Exibe o título de seleção com tamanho maior usando HTML
    st.sidebar.markdown(f"<h2 style='font-size: 24px;'>{texto['selecione_pagina']}</h2>", unsafe_allow_html=True)

    # Exibe as opções de navegação como Radio Button
    pagina = st.sidebar.radio(
        "",  # O label vazio, pois já colocamos o título personalizado acima
        [texto["pagina_principal"], texto["engenheiro"], texto["obras"], texto["artigos"], texto["videos"], texto["citações"], texto["programas"]]
    )

    # Exibe a página correspondente
    if pagina == texto["pagina_principal"]:
        pagina_principal(texto)
    elif pagina == texto["engenheiro"]:
        pagina_engenheiro(texto)
    elif pagina == texto["obras"]:
        pagina_obras(texto)
    elif pagina == texto["artigos"]:
        pagina_artigos()
    elif pagina == texto["programas"]:
        pagina_programas()
    elif pagina == texto["videos"]:
        pagina_videos()
    elif pagina == texto["citações"]:
        pagina_citacoes()

if __name__ == '__main__':
    main()
