import streamlit as st

# Função para mostrar a página de citações
def pagina_citacoes():
    # Código HTML contendo as citações
    citacoes_html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Artigos Relacionados a Fundações</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            h1 {
                text-align: center;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin-bottom: 20px;
            }
            a {
                text-decoration: none;
                color: #1a0dab;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>Artigos Relacionados a Fundações</h1>
        <ul>
            <li>
                <strong>Poulos, H. G. (2016).</strong> Tall building foundations: design methods and applications. Innovative Infrastructure Solutions, 1, 1-51.<br>
                <a href="https://link.springer.com/article/10.1007/s41062-016-0010-2" target="_blank">Acesse o artigo</a>
            </li>
            <li>
                <strong>Poulos, H. G. (1989).</strong> Pile behaviour—theory and application. Geotechnique, 39(3), 365-415.<br>
                <a href="https://www.icevirtuallibrary.com/doi/abs/10.1680/geot.1989.39.3.365" target="_blank">Acesse o artigo</a>
            </li>
            <li>
                <strong>Alonso, U. R. (2012).</strong> Dimensionamento de fundações profundas. Editora Blucher.<br>
                <a href="https://books.google.com/books?hl=pt-BR&lr=&id=7mKtDwAAQBAJ&oi=fnd&pg=PA1&ots=KY03q6shRG&sig=iTo0aEq506lGr2_0oI0KDuQg2oo" target="_blank">Acesse o livro</a>
            </li>
            <li>
                <strong>Fellenius, B. (2017).</strong> Basics of foundation design. Lulu. com.<br>
                <a href="https://books.google.com/books?hl=pt-BR&lr=&id=icVJDwAAQBAJ&oi=fnd&pg=PA1&ots=HITUhkMnAZ&sig=EyLrAWUdQTp6afbxT6GSnQ4hsSA" target="_blank">Acesse o livro</a>
            </li>
            <li>
                <strong>Poulos, H. G. (2017).</strong> Tall building foundation design. CRC Press.<br>
                <a href="https://www.taylorfrancis.com/books/mono/10.1201/9781315156071/tall-building-foundation-design-harry-poulos" target="_blank">Acesse o livro</a>
            </li>
            <li>
                <strong>Salgado, R. (2022).</strong> The engineering of foundations, slopes and retaining structures. CRC Press.<br>
                <a href="https://books.google.com/books?hl=pt-BR&lr=&id=LS5sEAAAQBAJ&oi=fnd&pg=PP1&ots=WgbJplGFn9&sig=3erqnPPzCdM70sQ4WZ4VEX5KFWA" target="_blank">Acesse o livro</a>
            </li>
            <li>
                <strong>Huat, C. Y., et al. (2021).</strong> Factors influencing pile friction bearing capacity: Proposing a novel procedure based on gradient boosted tree technique. Sustainability, 13(21), 11862.<br>
                <a href="https://www.mdpi.com/2071-1050/13/21/11862" target="_blank">Acesse o artigo</a>
            </li>
            <li>
                <strong>Al-Jeznawi, D., et al. (2024).</strong> Novel Explicit Models for Assessing the Frictional Resistance of Pipe Piles Subjected to Seismic Effects. Journal of Safety Science and Resilience.<br>
                <a href="https://www.sciencedirect.com/science/article/pii/S2666449624000537" target="_blank">Acesse o artigo</a>
            </li>
            <li>
                <strong>Ali, B. (2023).</strong> Contribution of the standard penetration test SPT to the design of pile foundations in sand–Practical recommendations. Journal of Engineering Research, 11(3).<br>
                <a href="https://www.researchgate.net/profile/Ali-Bouafia/publication/365469781_Contribution_of_the_standard_penetration_test_SPT_to_the_design_of_pile_foundations_in_sand-_Practical_recommendations/links/6380fdaf48124c2bc66c73db/Contribution-of-the-standard-penetration-test-SPT-to-the-design-of-pile-foundations-in-sand-Practical-recommendations.pdf" target="_blank">Acesse o artigo</a>
            </li>
            <li>
                <strong>Bui-Ngoc, T., et al. (2024).</strong> Predicting load–displacement of driven PHC pipe piles using stacking ensemble with Pareto optimization. Engineering Structures, 316, 118574.<br>
                <a href="https://www.sciencedirect.com/science/article/pii/S0141029624011362" target="_blank">Acesse o artigo</a>
            </li>
        </ul>
    </body>
    </html>
    """
    # Renderizar o HTML no Streamlit
    st.markdown(citacoes_html, unsafe_allow_html=True)

# Adicionar esta página na navegação do seu aplicativo
