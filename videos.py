import streamlit as st

def pagina_videos():
    st.title('Vídeos de Luciano Decourt')

    # Lista de vídeos
    videos = [
        "https://www.youtube.com/watch?v=MDH-3b3ZfXw&t=4s&pp=ygUPbHVjaWFubyBkZWNvdXJ0",
        "https://www.youtube.com/watch?v=XviWKYvEkTE&t=2s&pp=ygUPbHVjaWFubyBkZWNvdXJ0",
        "https://www.youtube.com/watch?v=wz3ESzvsTac&pp=ygUPbHVjaWFubyBkZWNvdXJ0",
        "https://www.youtube.com/watch?v=uGMoM5Kp44U&pp=ygUPbHVjaWFubyBkZWNvdXJ0",
        "https://www.youtube.com/watch?v=SXMCMYotRtg&pp=ygUPbHVjaWFubyBkZWNvdXJ0"
    ]

    # Exibe os vídeos na página
    for video in videos:
        st.video(video)

