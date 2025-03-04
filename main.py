import streamlit as st
import requests, whisper
from tools.get_transcript import get_transcript
from tools.process_audio import reduce_bruit

st.set_page_config(layout="wide")

# Définir l'URL de l'API (à adapter selon où tourne votre API FastAPI)
API_URL_R = "http://localhost:8000/get_resume/"
API_URL_D = "http://localhost:8000/get_diagnostic/"
API_URL_P = "http://localhost:8000/get_proposition/"

st.title(":blue[Health Assistant AI] :sunglasses:")
# st.write("Entrez une conversation pour interagir avec l'API.")

@st.cache_resource(ttl="2h")
@st.cache_data(ttl="2h")
def load_model():
     return whisper.load_model("turbo")

if "resume" not in st.session_state:
    st.session_state.resume = None
if "diagnostic" not in st.session_state:
    st.session_state.diagnostic = None
if "proposition" not in st.session_state:
    st.session_state.proposition = None
if "audio_value" not in st.session_state:
    st.session_state.audio_value = None
if "text_audio" not in st.session_state:
    st.session_state.text_audio = None
# if "model" not in st.session_state:
#     st.session_state.model = load_model()


st.session_state.audio_value = st.experimental_audio_input("Record a voice message")
# st.markdown(type(st.session_state.audio_value))
if st.session_state.audio_value:
    path = "audio.wav"
    with open(path, "wb") as file:
        file.write(st.session_state.audio_value.getbuffer())
    
    if st.button("process audio"):
        with st.spinner("processing audio file..."):
            reduce_bruit(path=r"C:\Users\PATEGOU\Downloads\test_audio.wav") #path

        st.write("audio process")
        try:
            st.audio(data=r"C:\Users\PATEGOU\Downloads\test_audio.wav") #f"{path}_reduit.wav"
        except Exception as e:
            st.write(f"errro Occured : \t\t{e}")
            pass
        
        with st.spinner("generate transcript..."): 
            # st.session_state.text_audio = get_transcript(audio=f"{path}_reduit.wav", model=st.session_state.model)
            st.session_state.text_audio = True #get_transcript(audio=r"C:\Users\PATEGOU\Downloads\test_audio.wav", model=st.session_state.model)
    
   

col1 , col2, col3 = st.columns(spec=3, vertical_alignment="top")

# Bouton pour envoyer la requête
with col1:
    st.header(":red[Conversation]")
    # Champ de texte pour l'entrée utilisateur
    # if st.session_state.text_audio:
    user_input = st.text_area("Conversation :", value="", height=300)
    if st.button("Summary"):
        if user_input:
            # Afficher un message d'attente
            with st.spinner("Synthèse de la conversation..."):
                try:
                    # Envoyer la requête POST à l'API avec la question de l'utilisateur
                    response = requests.post(API_URL_R, json={"question": user_input})

                    # Vérifier si la requête a réussi
                    if response.status_code == 200:
                        st.session_state.resume = response.json()
                        st.success("Réponse reçue avec succès!")
                        # st.json(result)  # Afficher la réponse en format JSON
                    else:
                        st.error(f"Erreur {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de la connexion à l'API: {e}")
            # Note d'information
            if st.session_state.resume is not None:
                st.markdown(
                    st.session_state.resume["response"]
                )
        else:
            st.info("Veuillez entrer une conversation avant de soumettre.")
    see_resume = st.toggle("see resume")

with col2:
    st.header(":green[Resume]")
    # Champ de texte pour l'entrée utilisateur
    if st.session_state.resume:
        user_input = st.text_area("Resume :", value=st.session_state.resume["response"], height=300)
        if st.button("Generate Diagnostic"):
            with st.spinner("Diagnostic..."):
                try:
                    # Envoyer la requête POST à l'API avec la question de l'utilisateur
                    response_ = requests.post(API_URL_D, json={"question": user_input})

                    # Vérifier si la requête a réussi
                    if response_.status_code == 200:
                        st.session_state.diagnostic = response_.json()
                        st.success("Réponse reçue avec succès!")
                        # st.json(result_)  # Afficher la réponse en format JSON
                    else:
                        st.error(f"Erreur {response_.status_code}: {response_.text}")
                except Exception as e:
                        st.error(f"Erreur lors de la connexion à l'API: {e}")

            if st.session_state.diagnostic is not None:
                st.markdown(st.session_state.diagnostic)
        else:
            st.info("Veuillez entrer une conversation avant de soumettre.")
        see_diagnostic = st.toggle("see diagnostic")

with col3:
    st.header(":orange[Diagnostic]")
    if st.session_state.diagnostic:
        ser_input = st.text_area(label="Diagnostic", value=st.session_state.diagnostic, height=300)
        if st.button("Generate Proposition"):
            with st.spinner("proposition..."):
                try:
                    # Envoyer la requête POST à l'API avec la question de l'utilisateur
                    response__ = requests.post(API_URL_P, json={"question": user_input})

                    # Vérifier si la requête a réussi
                    if response__.status_code == 200:
                        st.session_state.proposition = response__.json()
                        st.success("Réponse reçue avec succès!")
                        # st.json(result_)  # Afficher la réponse en format JSON
                    else:
                        st.error(f"Erreur {response__.status_code}: {response__.text}")
                except Exception as e:
                        st.error(f"Erreur lors de la connexion à l'API: {e}")

            if st.session_state.proposition is not None:
                st.markdown(st.session_state.proposition)
        else:
            st.info("Veuillez entrer une conversation avant de soumettre.")
        see_proposition = st.toggle("see proposition")


if st.session_state.resume is not None and see_resume:
            st.markdown(st.session_state.resume["response"])

if st.session_state.diagnostic is not None and see_diagnostic:
            st.markdown(st.session_state.diagnostic)

if st.session_state.proposition is not None and see_proposition:
            st.markdown(st.session_state.proposition)

