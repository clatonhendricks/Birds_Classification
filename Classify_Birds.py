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
    
    chart_data = pd.DataFrame(list(zip(Bird_name,Prb)), columns=["Bird Names","Percentage"]) 

    chart =alt.Chart(chart_data).mark_bar(color='black').encode(
        y=alt.Y('Bird Names:N',sort=alt.EncodingSortField(field="Percentage", op="sum", order='descending')),
        x=alt.X('Percentage:Q'),
        color=alt.Color('Bird Names:N', legend=None),
        )
    
    chart.configure_title(
    fontSize=20,
    font='Courier',
    anchor='middle',
    color='gray')

    text = chart.mark_text(
    align='left',
    baseline='middle',
    color='black',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
    text='Percentage:Q')

    st.write((chart+text).properties(title='Probabilities ratio',width=700, height=300))

def Intro_text():
    st.write('This detection model was created using FastAI deep learning framework using the '+
     '[CalTech 200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset. ' +

    'This dataset has 200 different north american bird species with a total of 11,700 images. Source code of this app is at my [github](https://github.com/clatonhendricks/Birds_Classification) repo. ' + 
    'Currently this model has a 80% accuracy rate. (Yes, I will get it in the higher 90s in a few days)'
                )


def main():
    img_banner = P_Image.open('Birds_AutoCollage.jpg')
    st.image(img_banner, use_column_width=True)
    st.title("Bird classification web app")
    Intro_text()
    img_path = st.file_uploader("Upload an image of a bird",type=['jpg','png'])


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
        
    st.write('List of birds supported by the model')
    df = pd.read_csv('BirdList.csv', header=None, names=['Bird Names'])
    st.write(df)
    
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    learn_inf = load_path(Path()/'stage-1-80.pkl')
    center_img()
    main()
