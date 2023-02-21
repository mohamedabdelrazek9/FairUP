import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Homepage",
    layout="wide"
)

# Create a text box
#text_box = st.text_input("Enter some text:")

# Create an expander to display additional information
#with st.expander("More information"):
#    st.write("This section contains additional information about the text box.")

# Display the text box
#st.write("You entered:", text_box)

ovgu_img2 = Image.open('ovgu_logo.png')
st.image(ovgu_img2)
st.title("FairUP: a Framework for Fairness Analysis of Graph Neural Network-Based User Profiling Models. 🚀")
ovgu_img = Image.open('fairup_architecture-1.png')
#ovgu_img = ovgu_img.resize((1000, 1000))
st.markdown('##### We have developed a comprehensive framework for Graph Neural Networks-based user profiling models that empowers researchers and users to simultaneously train multiple models and analyze their outcomes. This framework includes tools for mitigating bias, ensuring fairness, and increasing model interpretability. Our approach allows for the incorporation of debiasing techniques into the training process, which helps to minimize the impact of societal biases on model performance. In addition, our framework supports multiple evaluation metrics, enabling the user to compare and contrast the performance of different models.')
st.text("")
st.markdown('##### The overall structure of the framework is as follows:')
st.text("")
st.text("")
st.image(ovgu_img)
#st.text("")
#st.markdown('##### The framework is divided into 3 components: the Pre-processing component, the Core component, and the Post-processing fairness evaluation component')

#st.sidebar.success("Select a page")