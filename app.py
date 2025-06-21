import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model_one.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("<br>", unsafe_allow_html=True)
    image_path = "ali.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                
""")
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

# Bottom bar using HTML & CSS
    st.markdown("""
    <style>
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ccc;
        z-index: 9999;
    }
    </style>

    <div class="bottom-bar">
        © 2025 KAZAD PVT ltd — All rights reserved <br><B>For More details Contact us</B>   |     Contact No: 9449605263          |       email:<a>aliammarhasan2123@gmail.com</a>
    </div>
""", unsafe_allow_html=True)

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
    #### About Our Team
    We are a passionate team of five tech enthusiasts who came together with a shared goal: to harness the power of deep learning to support agriculture. Our project, Plant Disease Detection Using Deep Learning Algorithms, aims to help farmers and agricultural experts detect plant diseases early through an intelligent, image-based detection system.

    Meet the Team:
                
    [**Ali Ammar Hasan**] – **Team Leader & Deep Learning Engineer**
                
    Oversaw the project from start to finish and Built and fine-tuned the deep learning model using convolutional neural networks (CNNs) for accurate detection of plant diseases from leaf images.

    [**Almas Raza**] – **Dataset Manager & Preprocessing Expert**
                
    Collected and cleaned the image dataset, performed data augmentation, and ensured high-quality input for training the model.

    [**Disha Rathod**] – **Research & Documentation Lead**
                
    Focused on researching plant diseases, symptoms, and agricultural impacts, and prepared clear documentation and project reports.

    [**Ahmed Kaif**] – **UI/UX Designer & Presentation Specialist**
                
    Designed the interface for an intuitive user experience and contributed to the visual elements and final project presentation.

    [**Syed Zaheer Ul Haque**] – **Web Developer**
                
    developed the web application to make the detection tool accessible and user-friendly.

    Our teamwork reflects a blend of creativity, technology, and purpose. With this project, we hope to bridge the gap between artificial intelligence and sustainable agriculture.
                
    #### About Project 
    
    Agriculture plays a vital role in sustaining economies and ensuring food security worldwide. One of the major challenges faced by farmers is the early and accurate detection of plant diseases, which can severely affect crop quality and yield. Traditional methods of disease identification are time-consuming, labor-intensive, and often require expert knowledge.

    This project, "Plant Disease Detection Using Deep Learning Algorithms," presents an intelligent solution to this problem by leveraging the power of artificial intelligence and computer vision. We developed a deep learning-based model using Convolutional Neural Networks (CNNs) to automatically identify plant diseases from leaf images. The system was trained on a diverse dataset of diseased and healthy plant leaves, achieving high accuracy in classification.

    The final solution is integrated into a user-friendly web application that allows users to upload images of plant leaves and receive instant feedback about possible diseases, along with suggested preventive measures. This tool is designed to assist farmers, agricultural workers, and researchers in making timely decisions to reduce crop losses and enhance productivity.

    By combining deep learning with accessible technology, our project offers a scalable and cost-effective approach to disease detection in agriculture, contributing to more sustainable and informed farming practices.

""")
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

        