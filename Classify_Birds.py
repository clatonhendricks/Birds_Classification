import streamlit as st
from PIL import Image as P_Image
from fastai.vision.all import *
from pathlib import Path, PurePath, PureWindowsPath
import torch
import pathlib

def load_posix_learner(path):
    save = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    learn = load_learner(path)
    
    pathlib.PosixPath = save
    return learn


learn_inf = load_posix_learner(Path()/'export.pkl')
st.title("Birds classification app")
st.text("Select an image of a bird")

img_path = st.file_uploader("Upload an image",type=['jpg','png'])


if img_path:
    img = P_Image.open(img_path)
    img_array = np.array(img)
    st.image(img, use_column_width=True)
    pred, pred_idx, probs = learn_inf.predict(img_array)
    prob_pct = probs[pred_idx] * 100
    st.write('**Prediction**: ', pred, '**Probability:**: ',prob_pct )
    

