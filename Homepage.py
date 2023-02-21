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

logo_ovgu_fin = Image.open('imgs/logo_ovgu_fin_en.jpg')
st.image(logo_ovgu_fin)

st.title("FairUP: a Framework for Fairness Analysis of Graph Neural Network-Based User Profiling Models 🚀")
st.markdown("##### *Mohamed Abdelrazek, Erasmo Purificato, Ludovico Boratto, and Ernesto William De Luca*")

st.markdown("## Description")
st.markdown("""
**FairUP** is a standardised framework that empowers researchers and practitioners to simultaneously analyse state-of-the-art Graph Neural Network-based models for user profiling task, in terms of classification performance and fairness metrics scores.

The framework, whose architecture is shown below, presents several components, which allow end-users to:
* compute the fairness of the input dataset by means of a pre-processing fairness metric, i.e. *disparate impact*;
* mitigate the unfairness of the dataset, if needed, by applying different debiasing methods, i.e. *sampling*, *reweighting* and *disparate impact remover*; 
* standardise the input (a graph in Neo4J or NetworkX format) for each of the included GNNs;
* train one or more GNN models, specifying the parameters for each of them;
* evaluate post-hoc fairness by exploiting four metrics, i.e. *statistical parity*, *equal opportunity*, *overall accuracy equality*, *treatment equality*.
""")

# st.markdown('##### We have developed a comprehensive framework for Graph Neural Networks-based user profiling models that empowers researchers and users to simultaneously train multiple models and analyze their outcomes. This framework includes tools for mitigating bias, ensuring fairness, and increasing model interpretability. Our approach allows for the incorporation of debiasing techniques into the training process, which helps to minimize the impact of societal biases on model performance. In addition, our framework supports multiple evaluation metrics, enabling the user to compare and contrast the performance of different models.')

# Vertical space
st.text("")

fairup = Image.open('imgs/fairup_architecture.png')
st.image(fairup, caption="Logical architecture of FairUP framework")

#st.text("")
#st.markdown('##### The framework is divided into 3 components: the Pre-processing component, the Core component, and the Post-processing fairness evaluation component')

#st.sidebar.success("Select a page")