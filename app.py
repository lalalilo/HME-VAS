import cv2
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import sympy
from transcriptor import Transcriptor
from utils import get_bounding_boxes_yolov8, clean_latex_expression

title = "Reconnaissance de calculs posés à la vertical à l'aide d'OCR et d'IA"

st.set_page_config(layout="wide", page_title=title, page_icon="🤖", )

st.title(title)
st.header("1. Choisissons une image contenant une opération au hasard")


if st.button("Clique-ici 👀", ):
    random_number = np.random.randint(1, 305)
    image_path = f"dataset/images/{random_number}.png"
    st.header("2. Regardons l'image sélectionnée")
    picture = st.image(image_path)

    st.header("3. Affichons le résultat de la reconnaissance de l'opération et vérifions si elle est correcte 🤞")


    try:

        with st.spinner("Traitement en cours..."):
            time.sleep(1)

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
            st.image(picture)

            latex = transcriptor(bounding_boxes)
            latex = clean_latex_expression(latex)
            st.markdown("### Le calcul extrait de l'image est : ")
            with st.container(border=True):
                st.latex(latex.replace("==", "="))
            sympy.init_printing()
            sympy_expr = sympy.sympify(latex)
            if sympy_expr: 
                st.success("Ce qui est... correct !")
            else: 
                st.error('Ce qui est... incorrect !')
    except Exception:
        st.warning("Oups, nous avons rencontré une erreur 😰 Nous sommes encore en beta ! Essayez avec une autre image 🤞")





# # Specify canvas parameters in application
# drawing_mode = "freedraw"
# stroke_width = 3
# realtime_update = True

# # Create a canvas component
# canvas_result = st_canvas(
#     stroke_width=stroke_width,
#     stroke_color="#F31313",
#     update_streamlit=realtime_update,
#     drawing_mode=drawing_mode,
#     key="canvas",
# )

# # Do something interesting with the image data and paths
# if st.button("process"):
#     # Open the image file

#     img_data = canvas_result.image_data
#     if canvas_result.image_data is not None:
#         negative_img_data = 255 - canvas_result.image_data
#         negative_img = Image.fromarray(negative_img_data.astype(np.uint8)).convert(
#             "RGB"
#         )
#         negative_img.save("img.jpg", "JPEG")
#     else:
#         st.write("no image to save")

#     transcriptor = Transcriptor()

#     bounding_boxes = get_bounding_boxes_yolov8("img.jpg")
#     print(bounding_boxes)

#     latex = transcriptor(bounding_boxes)
#     st.markdown(latex)

#     evaluate_expression(latex)
