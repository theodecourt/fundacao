import streamlit as st

def mostrar_curriculo():
    # Carregar e exibir a imagem no lugar do título
    st.image("foto.jpeg", caption="Luciano Decourt", use_column_width=True)
    
    # Currículo com formatação HTML
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
