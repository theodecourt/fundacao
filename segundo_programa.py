import streamlit as st
import math

# Função para calcular s0
def calcular_s0(q, Beq, quc, J10, L10):
    P10 = (math.log10(q / (quc * J10)) - L10) / L10
    s0 = 10 ** P10 * Beq * 1000
    return s0

# Função para ajustar q para s1
def ajustar_q_para_s1(Q, s1, quc, J10, L10, tolerancia=0.01, max_iter=100):
    q_min = 0.01
    q_max = 10 * quc
    q_new = (q_max + q_min) / 2
    Beq = math.sqrt(Q / q_new)

    for _ in range(max_iter):
        s0 = calcular_s0(q_new, Beq, quc, J10, L10)
        if abs(s0 - s1) < tolerancia:
            return q_new
        if s0 < s1:
            q_min = q_new
        else:
            q_max = q_new
        q_new = (q_max + q_min) / 2
        Beq = math.sqrt(Q / q_new)

    return q_new

# Função para calcular os valores iniciais
def calcular_valores(Q, NSPT, tipo_solo):
    if tipo_solo == 1:
        quc = 12 * NSPT
    elif tipo_solo == 2:
        quc = 10 * NSPT
    else:
        quc = 8 * NSPT

    q = quc * 0.4
    Beq = math.sqrt(Q / q)
    area = Beq ** 2
    s0 = calcular_s0(q, Beq, quc, 1, 0.42)

    return quc, q, Beq, area, s0

# Função principal do segundo programa para ser exibido no Streamlit
def segundo_programa():
    st.title("Segundo Programa - Cálculos Geotécnicos")

    # Solicita os dados de entrada do usuário
    Q = st.number_input("Digite o valor de Q (tf):", min_value=0.0, value=100.0, key="Q")
    NSPT = st.number_input("Digite o valor de NSPT:", min_value=0.0, value=10.0, key="NSPT")
    tipo_solo = st.selectbox("Digite o tipo de solo (1, 2 ou 3):", [1, 2, 3], key="tipo_solo")

    # Inicializa os valores na session_state se ainda não estiverem salvos
    if "quc" not in st.session_state:
        if st.button("Calcular valores iniciais"):
            quc, q, Beq, area, s0 = calcular_valores(Q, NSPT, tipo_solo)
            # Salva os valores no session_state
            st.session_state["quc"] = quc
            st.session_state["q"] = q
            st.session_state["Beq"] = Beq
            st.session_state["area"] = area
            st.session_state["s0"] = s0

    if "quc" in st.session_state:
        # Exibe os resultados salvos
        st.write(f"quc (tf/m²): {st.session_state['quc']}")
        st.write(f"q (tf/m²): {st.session_state['q']}")
        st.write(f"Beq (m): {st.session_state['Beq']}")
        st.write(f"Área inicial (m²): {st.session_state['area']}")
        st.write(f"s0 inicial (mm): {st.session_state['s0']}")

        # Solicita o valor de s1 (mm)
        s1 = st.number_input("Digite o valor de s1 (mm):", min_value=0.0, value=10.0)

        if st.button("Ajustar q para s1"):
            # Calcula o novo q que resultaria em s0 = s1
            q_new = ajustar_q_para_s1(Q, s1, st.session_state["quc"], 1, 0.42)
            Beq_new = math.sqrt(Q / q_new)
            area_new = Beq_new ** 2

            # Exibe o resultado do novo q, Beq e área
            st.write(f"Novo q (tf/m²) para s0 = s1: {q_new}")
            st.write(f"Novo Beq (m) com q ajustado: {Beq_new}")
            st.write(f"Nova Área (m²) com Beq ajustado: {area_new}")
