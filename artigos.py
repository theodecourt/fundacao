import streamlit as st

def pagina_artigos(idioma):
    # Criação da página de artigos
    if idioma == "Português":
        st.title('Artigos de Luciano Décourt')
    else:
        st.title('Articles by Luciano Décourt')
    

    artigos_html = """
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

    <ul>
        <li>
            <strong>Décourt, L. (2021). Prediction of the bearing capacity of piles based exclusively on N values of the SPT.</strong> In Penetration Testing, volume 1 (pp. 29-34). Routledge.<br>
            <a href="https://api.taylorfrancis.com/content/chapters/edit/download?identifierName=doi&identifierValue=10.1201/9780203743959-4&type=chapterpdf" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Resende, A. S., Décourt, L., Silva, M. Q. D. C., Araújo, C. B. C. D., & Nóbrega Junior, A. J. (2019).</strong> Comparativo de resultados de provas de carga com células expansivas, ensaios bidirecionais, em estacas hélice contínua sem e com problemas em processo executivo.<br>
            <a href="https://repositorio.ufc.br/handle/riufc/59179" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2018). Design of shallow foundations on soils and rocks on basis of settlement considerations.</strong> In Innovations in geotechnical engineering (pp. 342-357).<br>
            <a href="https://ascelibrary.org/doi/abs/10.1061/9780784481639.023" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Abreu, P. S. B., Décourt, L., & de Souza Filho, J. M. (2015).</strong> Execução de Estacas em Solos Lareríticos. In From Fundamentals to Applications in Geotechnics (pp. 1568-1574). IOS Press.<br>
            <a href="https://ebooks.iospress.nl/doi/10.3233/978-1-61499-603-3-1568" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2015).</strong> Prediction of Bearing Capacity of Bored Piles in Sands. Theoretical x Empirical Formulas. In From Fundamentals to Applications in Geotechnics (pp. 1721-1725). IOS Press.<br>
            <a href="https://ebooks.iospress.nl/doi/10.3233/978-1-61499-603-3-1721" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L., & Koshima, A. (2015).</strong> Remedial Measures for a Building With Pile Foundations in Subsiding Soil. In From Fundamentals to Applications in Geotechnics (pp. 1670-1677). IOS Press.<br>
            <a href="https://ebooks.iospress.nl/doi/10.3233/978-1-61499-603-3-1670" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2013).</strong> Class A predictions and benefits derived from their analyses. In Geotechnical and Geophysical Site Characterization: Proceedings of the 4th International Conference on Site Characterization ISC-4 (Vol. 1, pp. 1797-1804). Taylor & Francis Books Ltd.<br>
            <a href="https://search.proquest.com/openview/64bc4fd66c62ac2e70907bc319d00e7a/1?pq-origsite=gscholar&cbl=2069208" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Melo, B. N., ALBUQUERQUE, J., Décourt, L., & Carvalho, D. (2012).</strong> Análise do atrito lateral em estacas hélice contínua instrumentadas por meio do conceito de Rigidez. In 16º Congresso Brasileiro de Mecânica dos Solos e Engenharia Geotécnica. Porto de Galinhas.<br>
            <a href="https://www.fec.unicamp.br/~pjra/wp-content/uploads/2020/01/28-An%C3%A1lise-do-atrito-lateral-em-estacas-h%C3%A9lice-cont%C3%ADnua-instrumentadas-por-meio-do-conceito-de-rigidez-%E2%80%93-COBRAMSEG-%E2%80%93-2012.pdf" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2011).</strong> Discussion of “Plate Load Tests on Cemented Soil Layers Overlaying Weaker Soil” by Nilo Cesar Consoli, Francisco Dalla Rosa, and Anderson Fonini. Journal of Geotechnical and Geoenvironmental Engineering, 137(4), 447-448.<br>
            <a href="https://ascelibrary.org/doi/abs/10.1061/(asce)gt.1943-5606.0000380" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2008).</strong> Loading tests: interpretation and prediction of their results. In From Research to Practice in Geotechnical Engineering (pp. 452-470).<br>
            <a href="https://ascelibrary.org/doi/abs/10.1061/40962(325)16" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2007).</strong> Discussion of “Pile Behavior—Consequences of Geological and Construction Imperfections” by Harry G. Poulos. Journal of Geotechnical and Geoenvironmental Engineering, 133(1), 120-120.<br>
            <a href="https://ascelibrary.org/doi/full/10.1061/(ASCE)1090-0241(2007)133:1(120)" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Décourt, L. (2006).</strong> Discussion of “Estimation of Bearing Capacity of Circular Footings on Sands Based on Cone Penetration Test” by Junhwan Lee and Rodrigo Salgado. Journal of Geotechnical and Geoenvironmental Engineering, 132(11), 1511-1513.<br>
            <a href="https://ascelibrary.org/doi/abs/10.1061/(ASCE)1090-0241(2006)132:11(1511)" target="_blank">Acesse o artigo</a>
        </li>
        <li>
            <strong>Decourt, L. (1989).</strong> Discussion of “Concrete Pile Design in Tidewater Virginia” by Ray E. Martin, James J. Seli, Graydon W. Powell, and Michael Bertoulin (June, 1987, Vol. 113, No. 6). Journal of Geotechnical Engineering, 115(10), 1499-1500.<br>
            <a href="https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9410(1989)115:10(1499)" target="_blank">Acesse o artigo</a>
        </li>
    </ul>
    """
    st.markdown(artigos_html, unsafe_allow_html=True)
