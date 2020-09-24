import streamlit as st
from PIL import Image as P_Image
from fastai.vision.all import *
import pathlib

def load_posix_learner(path):
    save = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    learn = load_learner(path)
    
    pathlib.PosixPath = save
    return learn

def progress(prog_text):
    with st.spinner(prog_text):
        time.sleep(5)

def main():
    st.title("Bird classification app")
    st.text("Select an image of a bird")

    img_path = st.file_uploader("Upload an image",type=['jpg','png'])


    if img_path:
        progress('Detecting. Please wait..')
        img = P_Image.open(img_path)
        img_array = np.array(img)
        st.image(img, use_column_width=True)
        pred, pred_idx, probs = learn_inf.predict(img_array)
        prob_pct = probs[pred_idx] * 100
        st.write('**Prediction**: ', pred, '**Probability:**: ',round(prob_pct.item(),3))

if __name__ == '__main__':
    learn_inf = load_posix_learner(Path()/'export.pkl')
    main()
