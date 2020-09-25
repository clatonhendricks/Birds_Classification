import streamlit as st
from PIL import Image as P_Image
from fastai.vision.all import *
import pathlib
import platform
import pandas as pd

def load_path(path):
    save = pathlib.PosixPath
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.PosiPath = pathlib.PurePath
    
    learn = load_learner(path)
    
    pathlib.PosixPath = save
    return learn

def progress(prog_text):
    with st.spinner(prog_text):
        time.sleep(5)

def center_img():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def main():
    img_banner = P_Image.open('Birds_AutoCollage.jpg')
    st.image(img_banner, use_column_width=True)
    st.title("Bird classification app")
    st.text("Select an image of a bird")

    img_path = st.file_uploader("Upload an image",type=['jpg','png'])


    if img_path:
        img = P_Image.open(img_path)
        img_array = np.array(img)
        st.image(img, width=500)
        progress('Detecting. Please wait..')
        pred, pred_idx, probs = learn_inf.predict(img_array)
        prob_pct = probs[pred_idx] * 100
        center_img()
        st.write('**Prediction**: ', pred)
        st.write('**Probability %:** ',round(prob_pct.item(),3))



if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    learn_inf = load_path(Path()/'export.pkl')
    center_img()
    main()
