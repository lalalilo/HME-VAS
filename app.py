import cv2
import streamlit as st
import sympy
from transcriptor import Transcriptor
from utils import get_bounding_boxes_yolov8, clean_latex_expression, MDP

title = "Reconnaissance de calculs posés à la verticale à l'aide d'OCR et d'IA"
st.set_page_config(page_title=title, page_icon="🤖", layout="wide")


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
                Cette application permet de reconnaître des calculs posés à la verticale d’additions et de soustractions à l'aide d'OCR et d'IA.
                Elle est basée sur un modèle YOLOv5, entrainé spécifiquement sur un jeu de données de calculs posés à la verticale par des élèves.
                Le jeu de données contient des calculs corrects et des calculs erronés."""
            )
        with st.container(border=True):
            st.markdown(
                """
                Le modèle de *vision par ordinateur* est capable de détecter les particularités de la pose verticale de calculs : 
                alignement des chiffres et structure du calcul, utilisation du signe égal, utilisation ou non de retenues, etc."""
            )
        with st.container(border=True):
            st.markdown(
                """
                Sur cette démonstration, le modèle évalue en direct des calculs posés à la verticale et donne ensuite une évaluation du résultat.
                Quelques images sont données en exemple pour tester le modèle.
                """
            )

    st.image(image="ressources/BannerDemoMaths.svg", use_column_width="always")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1.container(border=True):
        st.markdown("#### Étape 1: affichage de l'image")

        demo_img_ref = [190, 73, 50, 140, 171, 3, 18, 12]

        if "current_image" not in st.session_state:
            st.session_state.current_image = 0

        # Load the initial image
        image_path = (
            f"dataset/images/{demo_img_ref[st.session_state.current_image]}.png"
        )
        picture = st.image(image_path, use_column_width="auto")

        with c2.container(border=True):
            st.markdown("#### Étape 2: reconnaissance de l'opération ")
            transcriptor = Transcriptor()
            if image_path:
                bounding_boxes = get_bounding_boxes_yolov8(image_path)

                # Load the image
                picture = cv2.imread(image_path)
                # Draw the bounding boxes on the image
                for box in bounding_boxes:
                    top_left, bottom_right = box[0]
                    label = box[1]
                    # conf = box[2]
                    cv2.rectangle(picture, top_left, bottom_right, (0, 0, 255), 1)
                    cv2.putText(
                        picture,
                        f"{label}",
                        top_left,
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 0),
                        1,
                    )

                # Plot the image
                st.image(
                    picture,
                    use_column_width="auto",
                    channels="BGR",
                )

                latex = transcriptor(bounding_boxes)
                latex = clean_latex_expression(latex)
        with c3.container(border=True):
            st.markdown("#### Étape 3: évaluation de l'opération")
            st.markdown("**Le calcul extrait de l'image est :** ")
            st.latex(latex.replace("==", "="))
            sympy.init_printing()
            sympy_expr = sympy.sympify(latex)
            if sympy_expr:
                st.success("**Le résultat de l'opération est correct** ✅")
            else:
                st.error(f"**Le résultat de l'opération est incorrect. La bonne réponse est {sympy.sympify(latex.split('=')[0])}.**")
        with c3.container(border=True):
            st.markdown("##### Essayer avec une autre image")
            if st.button("Charger une autre image", type="primary"):
                # Increment current_image and wrap around if it exceeds the list length
                st.session_state.current_image = (
                    st.session_state.current_image + 1
                ) % len(demo_img_ref)

                # Load the new image
                image_path = (
                    f"dataset/images/{demo_img_ref[st.session_state.current_image]}.png"
                )


def login():
    st.title("Connexion")
    password = st.text_input("Mot de passe", type="password", placeholder="Entrer le mot de passe", label_visibility="collapsed")

    if st.button("Se connecter"):
        if password == MDP:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main()
else:
    login()
