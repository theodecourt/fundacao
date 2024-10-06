import streamlit as st

# Função para mostrar o currículo com base no idioma selecionado
def mostrar_curriculo(idioma):
    if idioma == "Português":
        # Currículo em Português
        return """
        <div style="text-align: justify; font-size:16px;">
        <h4>Formação Profissional:</h4>
        <ul>
            <li>Escola Politécnica da Universidade de São Paulo, POLI, USP (1963)</li>
            <li>Harvard University (1965)</li>
            <li>Cambridge University (1972)</li>
        </ul>

        <h4>Empresas onde atuou e/ou atua:</h4>
        <ul>
            <li>Instituto de Pesquisas Técnológicas (IPT), Seção de Solos</li>
            <li>Brasconsult, Engenharia de Projetos</li>
            <li>Laboratório Rankine de Engenharia Civil</li>
            <li>Luciano Décourt Consultoria</li>
            <li>Faculdade de Engenharia da FAAP (Professor Titular de Mecânica dos Solos)</li>
        </ul>

        <h4>Entidades e Premiações:</h4>
        <ul>
            <li>Vice-presidente da International Society for Soil Mechanics and Foundation Engineering (1989 – 1994)</li>
            <li>Sócio Emérito da Associação Brasileira de Mecânica dos Solos e Engenharia Geotécnica (ABMS)</li>
            <li>Fellow da American Society of Civil Engineers (ASCE)</li>
            <li>Membro da Academia Nacional de Engenharia (ANE)</li>
        </ul>

        <h4>Prêmios:</h4>
        <ul>
            <li>Manuel Rocha, Karl Terzaghi, José Machado (ABMS)</li>
            <li>Odair Grillo (ABEF)</li>
            <li>Milton Vargas (Revista Fundações e Obras Geotécnicas)</li>
        </ul>

        <h4>Contribuições:</h4>
        <p>Mais de 150 trabalhos publicados</p>

        <h4>Destaques:</h4>
        <ul>
            <li>Desenvolvimento do Standard Penetration Test (SPT), International Reference Test Procedure (1988/1989)</li>
            <li>Estudos pioneiros sobre solos lateríticos nos últimos 30 anos</li>
        </ul>
        </div>
        """
    elif idioma == "English":
        # Currículo em Inglês
        return """
        <div style="text-align: justify; font-size:16px;">
        <h4>Professional Education:</h4>
        <ul>
            <li>Polytechnic School of the University of São Paulo, POLI, USP (1963)</li>
            <li>Harvard University (1965)</li>
            <li>Cambridge University (1972)</li>
        </ul>

        <h4>Companies where he worked or still works:</h4>
        <ul>
            <li>Institute of Technological Research (IPT), Soil Section</li>
            <li>Brasconsult, Engineering Projects</li>
            <li>Rankine Civil Engineering Laboratory</li>
            <li>Luciano Décourt Consultancy</li>
            <li>School of Engineering at FAAP (Professor of Soil Mechanics)</li>
        </ul>

        <h4>Entities and Awards:</h4>
        <ul>
            <li>Vice-president of the International Society for Soil Mechanics and Foundation Engineering (1989 – 1994)</li>
            <li>Honorary Member of the Brazilian Association of Soil Mechanics and Geotechnical Engineering (ABMS)</li>
            <li>Fellow of the American Society of Civil Engineers (ASCE)</li>
            <li>Member of the National Academy of Engineering (ANE)</li>
        </ul>

        <h4>Awards:</h4>
        <ul>
            <li>Manuel Rocha, Karl Terzaghi, José Machado (ABMS)</li>
            <li>Odair Grillo (ABEF)</li>
            <li>Milton Vargas (Foundations and Geotechnical Works Magazine)</li>
        </ul>

        <h4>Contributions:</h4>
        <p>More than 150 published works</p>

        <h4>Highlights:</h4>
        <ul>
            <li>Development of the Standard Penetration Test (SPT), International Reference Test Procedure (1988/1989)</li>
            <li>Pioneering studies on lateritic soils over the last 30 years</li>
        </ul>
        </div>
        """

# Função principal de navegação
def main():
    idioma = st.sidebar.selectbox("Choose Language", ["Português", "English"])

    if idioma == "Português":
        st.title("Luciano Decourt")
    else:
        st.title("Luciano Decourt")

    st.markdown(mostrar_curriculo(idioma), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
