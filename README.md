# Birds Classification Web app project

The intention of repo is to test my skills using FastAI and Streamlit. 
I built a Bird detection model using [CalTech 200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset. 

This dataset consists of 200 different north american bird species with a total of 11,700 images. The final outcome is to get up 90% prediction rate but am not there yet. The current model has 80% prediction rate which is pretty good using simple FastAI technique.

I built a simple web app using [Streamlit](https://www.streamlit.io/) framework to test & play around with the model and used [Heroku](https://www.heroku.com) to host and upload the app. 

The web app is very simple, just upload any north american birth and it predict the species of the bird along with a next highest probablity of the species
![Webpart1](https://user-images.githubusercontent.com/250326/95006372-c9a3ab00-05b8-11eb-8ac9-a2b128b69cea.jpg)
![Webpart2](https://user-images.githubusercontent.com/250326/95006430-4afb3d80-05b9-11eb-823d-1b481f5645ca.jpg)

Feel free to play around the project which I hosted here https://classify-birds-app.herokuapp.com/

I plan on uploading the jupyter notebook for the FastAI prediction model in my repo soon. 



 



