import cv2
import streamlit as st
import sympy
from transcriptor import Transcriptor
from utils import get_bounding_boxes_yolov8, clean_latex_expression, MDP

title = "Reconnaissance de calculs posés à la verticale à l'aide d'OCR et d'IA"
st.set_page_config(page_title=title, page_icon="🤖", layout='wide')

def main():
    with st.sidebar:
        st.image(
            "ressources/lilo.png",
            use_column_width="auto",
        )

        st.title("À propos")

        with st.container(border=True):
            st.markdown(
                """
                Cette application permet de reconnaître des calculs posés à la verticale à l'aide d'OCR et d'IA.
                Elle est basée sur un modèle YOLOv5 pré-entraîné, entrainé spécifiquement sur un jeu de données de calculs posés à la verticale par des élèves.
                Le jeu de données contient des calculs corrects et des calculs erronés.""")
        with st.container(border=True):
            st.markdown("""
                Le modèle de *computer vision* est capable de détecter les particularités de la pose verticale d'élèves : 
                alignement des chiffres et structure du calcul, utilisation du signe égal, utilisation ou non de retenues, etc.""")
        with st.container(border=True):
            st.markdown("""
                Sur cette démonstration, le modèle évalue en direct des calculs posés à la verticale et donne ensuite une évaluation du résultat.
                Quelques images sont données en exemple pour tester le modèle.
                """
        )


    st.title(title)

    st.image(image="ressources/lalilo.png", use_column_width="always")
    c1, c2 = st.columns(2)
    c1.markdown("### 1. Choix d'une image contenant une opération")

    demo_img_ref = [3,18,12,190,73, 50, 140, 171,]

    # Initialize current_image if it doesn't exist
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0

    if st.button("Cliquer ici pour charger une image", type="primary" ):
        image_path = f"dataset/images/{demo_img_ref[st.session_state.current_image]}.png"
        picture = c1.image(image_path, width=400)
        # Increment current_image and wrap around if it exceeds the list length
        st.session_state.current_image = (st.session_state.current_image + 1) % len(demo_img_ref)

        c2.markdown("### 2. Résultat de la reconnaissance de l'opération ")

        try:
            with st.spinner("Traitement en cours..."):
                transcriptor = Transcriptor()
                bounding_boxes = get_bounding_boxes_yolov8(image_path)

                # Load the image
                picture = cv2.imread(image_path)

                # Draw the bounding boxes on the image
                for box in bounding_boxes:
                    top_left, bottom_right = box[0]
                    label = box[1]
                    conf = box[2]
                    cv2.rectangle(picture, top_left, bottom_right, (255, 0, 0), 2)
                    cv2.putText(picture, f"{label} ({conf:.2f})", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Plot the image
                c2.image(picture, width=400, channels="RGB")

                latex = transcriptor(bounding_boxes)
                latex = clean_latex_expression(latex)
                
                st.markdown("### 3. Évaluation de l'opération")
                c3, c4 = st.columns(2)

                with c3.container(border=True):
                    st.markdown("Le calcul extrait de l'image est : ")
                    st.latex(latex.replace("==", "="))
                sympy.init_printing()
                sympy_expr = sympy.sympify(latex)
                if sympy_expr: 
                    c4.success("Le résultat de l'opération est correct", icon="✅")
                else: 
                    c4.error("Le résultat de l'opération est incorrect", icon="❌")
        except Exception:
            c4.warning("Oups, nous avons rencontré une erreur 😰 Nous sommes encore en beta ! Essayez avec une autre image 🤞")


def login():
    st.title("Connexion")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if password == MDP:
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.error("Mot de passe incorrect")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if st.session_state['authenticated']:
    main()
else:
    login()

