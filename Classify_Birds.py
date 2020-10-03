import streamlit as st
from PIL import Image as P_Image
from fastai.vision.all import *
import pathlib
import platform
import pandas as pd
import numpy
import altair as alt

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
        time.sleep(3)

def center_img():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def create_chart(model, model_prob):
    val, idx = model_prob.topk(5)
    i = 0
    Bird_name = []
    Prb = []
    for x in idx:
        Bird_name.append(model.dls.vocab[x.item()])
        Prb.append(round(val[i].item(), 4)*100)
        i=i+1
    
    chart_data = pd.DataFrame(list(zip(Bird_name,Prb)), columns=["Bird Names","Percentage"]) # Prb, index=[Bird_name], columns=['Percent %']) #
    st.write(alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y('Bird Names:N',sort=alt.EncodingSortField(field="Percentage", op="sum", order='descending')),
        x=alt.X('Percentage:Q'),
         color='Percentage:Q',
        )
       .properties(width=700, height=300))

def main():
    img_banner = P_Image.open('Birds_AutoCollage.jpg')
    st.image(img_banner, use_column_width=True)
    st.title("Bird classification app")
    st.text("This model was created using CalTech 200-2011 dataset using FastAI library.")

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
        create_chart(learn_inf,probs)

        # chart_data1 = pd.DataFrame(
        #     np.random.randn(10, 3),
        #     columns=["a", "b", "c"])
        # st.bar_chart(chart_data1)
        



if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    learn_inf = load_path(Path()/'stage-1-80.pkl')
    center_img()
    main()
