import streamlit as st
from PIL import Image
import time
import pandas as pd
import os 
import paramiko
import threading
import queue
import warnings
import re
import subprocess

st.set_page_config(layout="wide")
ovgu_img = Image.open('imgs/logo_ovgu_fin_en.jpg')
st.image(ovgu_img)
st.title("FairUP: a Framework for Fairness Analysis of Graph Neural Network-Based User Profiling Models. 🚀")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


nba_columns = ['user_id', 'SALARY', 'AGE', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA',
       '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
       'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF_x', 'POINTS', 'GP', 'MPG',
       'ORPM', 'DRPM', 'RPM', 'WINS_RPM', 'PIE', 'PACE', 'W', 'player_height',
       'player_weight', 'country', 'C', 'PF_y', 'PF-C', 'PG', 'SF', 'SG',
       'ATL', 'ATL/CLE', 'ATL/LAL', 'BKN', 'BKN/WSH', 'BOS', 'CHA', 'CHI',
       'CHI/OKC', 'CLE', 'CLE/DAL', 'CLE/MIA', 'DAL', 'DAL/BKN', 'DAL/PHI',
       'DEN', 'DEN/CHA', 'DEN/POR', 'DET', 'GS', 'GS/CHA', 'GS/SAC', 'HOU',
       'HOU/LAL', 'HOU/MEM', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL',
       'MIL/CHA', 'MIN', 'NO', 'NO/DAL', 'NO/MEM', 'NO/MIL', 'NO/MIN/SAC',
       'NO/ORL', 'NO/SAC', 'NY', 'NY/PHI', 'OKC', 'ORL', 'ORL/TOR', 'PHI',
       'PHI/OKC', 'PHX', 'POR', 'SA', 'SAC', 'TOR', 'UTAH', 'WSH']

pokec_columns = ['user_id',
 'public',
 'completion_percentage',
 'gender',
 'region',
 'AGE',
 'I_am_working_in_field',
 'spoken_languages_indicator',
 'anglicky',
 'nemecky',
 'rusky',
 'francuzsky',
 'spanielsky',
 'taliansky',
 'slovensky',
 'japonsky',
 'hobbies_indicator',
 'priatelia',
 'sportovanie',
 'pocuvanie hudby',
 'pozeranie filmov',
 'spanie',
 'kupalisko',
 'party',
 'cestovanie',
 'kino',
 'diskoteky',
 'nakupovanie',
 'tancovanie',
 'turistika',
 'surfovanie po webe',
 'praca s pc',
 'sex',
 'pc hry',
 'stanovanie',
 'varenie',
 'jedlo',
 'fotografovanie',
 'citanie',
 'malovanie',
 'chovatelstvo',
 'domace prace',
 'divadlo',
 'prace okolo domu',
 'prace v zahrade',
 'chodenie do muzei',
 'zberatelstvo',
 'hackovanie',
 'I_most_enjoy_good_food_indicator',
 'pri telke',
 'v dobrej restauracii',
 'pri svieckach s partnerom',
 'v posteli',
 'v prirode',
 'z partnerovho bruska',
 'v kuchyni pri stole',
 'pets_indicator',
 'pes',
 'mam psa',
 'nemam ziadne',
 'macka',
 'rybky',
 'mam macku',
 'mam rybky',
 'vtacik',
 'body_type_indicator',
 'priemerna',
 'vysportovana',
 'chuda',
 'velka a pekna',
 'tak trosku pri sebe',
 'eye_color_indicator',
 'hnede',
 'modre',
 'zelene',
 'hair_color_indicator',
 'cierne',
 'blond',
 'plave',
 'hair_type_indicator',
 'kratke',
 'dlhe',
 'rovne',
 'po plecia',
 'kucerave',
 'na jezka',
 'completed_level_of_education_indicator',
 'stredoskolske',
 'zakladne',
 'vysokoskolske',
 'ucnovske',
 'favourite_color_indicator',
 'modra',
 'cierna',
 'cervena',
 'biela',
 'zelena',
 'fialova',
 'zlta',
 'ruzova',
 'oranzova',
 'hneda',
 'relation_to_smoking_indicator',
 'nefajcim',
 'fajcim pravidelne',
 'fajcim prilezitostne',
 'uz nefajcim',
 'relation_to_alcohol_indicator',
 'pijem prilezitostne',
 'abstinent',
 'nepijem',
 'on_pokec_i_am_looking_for_indicator',
 'dobreho priatela',
 'priatelku',
 'niekoho na chatovanie',
 'udrzujem vztahy s priatelmi',
 'vaznu znamost',
 'sexualneho partnera',
 'dlhodoby seriozny vztah',
 'love_is_for_me_indicator',
 'nie je nic lepsie',
 'ako byt zamilovany(a)',
 'v laske vidim zmysel zivota',
 'v laske som sa sklamal(a)',
 'preto som velmi opatrny(a)',
 'laska je zakladom vyrovnaneho sexualneho zivota',
 'romanticka laska nie je pre mna',
 'davam prednost realite',
 'relation_to_casual_sex_indicator',
 'nedokazem mat s niekym sex bez lasky',
 'to skutocne zalezi len na okolnostiach',
 'sex mozem mat iba s niekym',
 'koho dobre poznam',
 'dokazem mat sex s kymkolvek',
 'kto dobre vyzera',
 'my_partner_should_be_indicator',
 'mojou chybajucou polovickou',
 'laskou mojho zivota',
 'moj najlepsi priatel',
 'absolutne zodpovedny a spolahlivy',
 'hlavne spolocensky typ',
 'clovek',
 'ktoreho uplne respektujem',
 'hlavne dobry milenec',
 'niekto',
 'marital_status_indicator',
 'slobodny(a)',
 'mam vazny vztah',
 'zenaty (vydata)',
 'rozvedeny(a)',
 'slobodny',
 'relation_to_children_indicator',
 'v buducnosti chcem mat deti',
 'I_like_movies_indicator',
 'komedie',
 'akcne',
 'horory',
 'serialy',
 'romanticke',
 'rodinne',
 'sci-fi',
 'historicke',
 'vojnove',
 'zahadne',
 'mysteriozne',
 'dokumentarne',
 'eroticke',
 'dramy',
 'fantasy',
 'muzikaly',
 'kasove trhaky',
 'umelecke',
 'alternativne',
 'I_like_watching_movie_indicator',
 'doma z gauca',
 'v kine',
 'u priatela',
 'priatelky',
 'I_like_music_indicator',
 'disko',
 'pop',
 'rock',
 'rap',
 'techno',
 'house',
 'hitparadovky',
 'sladaky',
 'hip-hop',
 'metal',
 'soundtracky',
 'punk',
 'oldies',
 'folklor a ludovky',
 'folk a country',
 'jazz',
 'klasicka hudba',
 'opery',
 'alternativa',
 'trance',
 'I_mostly_like_listening_to_music_indicator',
 'kedykolvek a kdekolvek',
 'na posteli',
 'pri chodzi',
 'na dobru noc',
 'na diskoteke',
 's partnerom',
 'vo vani',
 'v aute',
 'na koncerte',
 'pri sexe',
 'v praci',
 'the_idea_of_good_evening_indicator',
 'pozerat dobry film v tv',
 'pocuvat dobru hudbu',
 's kamaratmi do baru',
 'ist do kina alebo divadla',
 'surfovat na sieti a chatovat',
 'ist na koncert',
 'citat dobru knihu',
 'nieco dobre uvarit',
 'zhasnut svetla a meditovat',
 'ist do posilnovne',
 'I_like_specialties_from_kitchen_indicator',
 'slovenskej',
 'talianskej',
 'cinskej',
 'mexickej',
 'francuzskej',
 'greckej',
 'morske zivocichy',
 'vegetarianskej',
 'japonskej',
 'indickej',
 'I_am_going_to_concerts_indicator',
 'ja na koncerty nechodim',
 'zriedkavo',
 'my_active_sports_indicator',
 'plavanie',
 'futbal',
 'kolieskove korcule',
 'lyzovanie',
 'korculovanie',
 'behanie',
 'posilnovanie',
 'tenis',
 'hokej',
 'basketbal',
 'snowboarding',
 'pingpong',
 'auto-moto sporty',
 'bedminton',
 'volejbal',
 'aerobik',
 'bojove sporty',
 'hadzana',
 'skateboarding',
 'my_passive_sports_indicator',
 'baseball',
 'golf',
 'horolezectvo',
 'bezkovanie',
 'surfing',
 'I_like_books_indicator',
 'necitam knihy',
 'o zabave',
 'humor',
 'hry',
 'historicke romany',
 'rozpravky',
 'odbornu literaturu',
 'psychologicku literaturu',
 'literaturu pre rozvoj osobnosti',
 'cestopisy',
 'literaturu faktu',
 'poeziu',
 'zivotopisne a pamate',
 'pocitacovu literaturu',
 'filozoficku literaturu',
 'literaturu o umeni a architekture']

alibaba_columns = ['userid', 'final_gender_code', 'age_level', 'pvalue_level', 'occupation', 'new_user_class_level ', 'adgroup_id', 'clk', 'cate_id']
jd_columns = ['user_id',
 'gender',
 'age_range',
 'item_id',
 'cid1',
 'cid2',
 'cid3',
 'cid1_name',
 'cid2_name',
 'cid3_name',
 'brand_code',
 'price',
 'item_name',
 'seg_name']

# select moodel

#dataset = st.selectbox("Select which dataset you want to train on", ("NBA", "Pokec", "Alibaba", "JD"))  
dataset = st.selectbox("Which dataset do you want to evaluate?", ("NBA", "Pokec-z", "Alibaba", "JD"))
if dataset == "NBA":
    dataset = 'nba'
    predict_attr = st.selectbox("Select prediction label", nba_columns)
    sens_attr = st.selectbox("Select sensitive attribute", nba_columns)
elif dataset == "Pokec-z":
    dataset = 'pokec_z'
    predict_attr = st.selectbox("Select prediction label", pokec_columns)
    sens_attr = st.selectbox("Select sensitive attribute", pokec_columns)
elif dataset == "Alibaba":
    dataset = 'alibaba'
    predict_attr = st.selectbox("Select prediction label", alibaba_columns)
    sens_attr = st.selectbox("Select sensitive attribute", alibaba_columns)
elif dataset == 'JD':
    dataset = 'tecent'
    predict_attr = st.selectbox("Select prediction label", jd_columns)
    sens_attr = st.selectbox("Select sensitive attribute", jd_columns)


# todo get all columns of the selected dataset and change this to a selectbox
#predict_attr = st.text_input("Enter the prediction label")
#sens_attr = st.text_input("Enter the senstive attribute")
def read_output(stdout, queue):
    for line in stdout:
        queue.put(line.strip())

def execute_command_fairness(dataset, sens_attr, predict_attr):
    with st.spinner("Loading..."):
        time.sleep(1)
        #ssh = paramiko.SSHClient()
        # Automatically add the server's host key (for the first connection only)
        #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote server
        #ssh.connect('141.44.31.206', username='abdelrazek', password='Mohamed')
        
        #if dataset == 'nba':
        #    stdin_new, stdout_new, stderr_new = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 main.py --calc_fairness True --dataset_name {} --dataset_path ../nba.csv --special_case True --sens_attr {} --predict_attr {} --type 1'.format(dataset, sens_attr, predict_attr), get_pty=True)
        #elif dataset == 'alibaba':
        #    stdin_new, stdout_new, stderr_new = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 main.py --calc_fairness True --dataset_name {} --dataset_path ../alibaba_small.csv --special_case True --sens_attr {} --predict_attr {} --type 1'.format(dataset, sens_attr, predict_attr), get_pty=True)
        #elif dataset == 'tecent':
        #    stdin_new, stdout_new, stderr_new = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 main.py --calc_fairness True --dataset_name {} --dataset_path ../JD_small.csv --special_case True --sens_attr {} --predict_attr {} --type 1'.format(dataset, sens_attr, predict_attr), get_pty=True)
        #elif dataset == 'pokec_z':
        #    stdin_new, stdout_new, stderr_new = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 main.py --calc_fairness True --dataset_name {} --dataset_path ../Master-Thesis-dev/region_job.csv --special_case True --sens_attr {} --predict_attr {} --type 1'.format(dataset, sens_attr, predict_attr), get_pty=True)
        #output_queue = queue.Queue()
        # start a thread to continuously read the output from the stdout object
        test = 'pwd'
        st.text(os.system(test))
        output_thread = threading.Thread(target=read_output, args=(stderr_new, output_queue))
        output_thread.start()

        # display the output in the Streamlit UI
        while True:
            try:
                line = output_queue.get_nowait()
                st.text(line)
            except queue.Empty:
                if output_thread.is_alive():
                    continue
                else:
                    break

        # wait for the thread to finish
        output_thread.join()
        # print the output to the console
        for line in stdout_new:
            print(line.strip())
            if "Dataset" in line:
                st.text(line.strip())  
        ssh.close()

fairness_evaluation = st.radio("Do you want to evaluate the dataset fairness?", ("No", "Yes"))
with st.expander("More information"):
        st.write("Evaluate how fair the dataset, namely how much bias is affecting the dataset as a whole using the disparate impact metric.")
if fairness_evaluation == "Yes":
    if st.button('Calculate Fairness'):
    # todo send command to server to compute fairness
    # then show fairness
    # add info box
    #dataset_fairness = st.write('Dataset Fairness: 1.57 (Fair)') 
        #execute_command_fairness(dataset, sens_attr, predict_attr)
        commands = os.popen('cd src && python main.py --calc_fairness True --dataset_name nba --dataset_path ./datasets/NBA/nba.csv --special_case True --sens_attr country --predict_attr SALARY --type 1').read()
        #output = os.popen('cd')
        #output = os.popen('python main.py --calc_fairness True --dataset_name nba --dataset_path ./datasets/NBA/nba.csv --special_case True --sens_attr country --predict_attr SALARY --type 1').read()
        #st.text(output)
        print(commands)
    

#####################
debias = st.radio("Do you want to apply debias approaches?", ("No", "Yes"))
if "Yes" in debias:
    debias_approach = st.selectbox("Select which debias approach you want to apply", ["Sample", "Reweighting", "Disparate remover impact"])
    with st.expander("More information"):
        st.write("You can mitigate the bias using three pre-processing debaising approaches:")
        st.write("Sampling: Generates more data to overcome the bias between the different sensitive attributes and classes.")
        st.write("Reweighting Minimizing the bias in the dataset by assiging different weights to dataset tuples, for example giving the unfavorable sensntive attributes higher weights than favorable sensitive attributes")
        st.write("Disparate impact remover: Transforms the sensitive attribute features in a way that the correlation between the sensitive attribute features and the prediction class is reduced")




#if dataset != None:
#st.markdown("#### Select dataset")
#uploaded_file = st.file_uploader("Select dataset")
    #dataset_path = st.text_input("", value="")

model_type = st.multiselect("Select the models you want to train", ["FairGNN", "RHGN", "CatGCN"])

if "RHGN" in model_type and "FairGNN" in model_type:
    st.markdown("### Enter the general parameters")
    seed = st.number_input("Enter the prefered seed number", value=0)
    
    #predict_attr = st.text_input("Enter the prediction label")
    #sens_attr = st.text_input("Enter the senstive attribute")


    st.markdown("### Enter the RHGN parameters")
    num_hidden = st.text_input("Enter the number of hidden layers", value=0)
    with st.expander("More information"):
        st.write("The number of hidden layers refers to the number of layers between the input layer and the output layer of a model.")
    lr_rhgn = st.number_input("Enter the learning rate for RHGN")
    with st.expander("More information"):
        st.write("Is a hyperparameter that controls the step size of the updates made to the weights during training. In other words, it determines how quickly the model learns from the data.")
    
    epochs_rhgn = st.number_input("Enter the number of epochs for RHGN", value=0)
    with st.expander("More information"):
        st.write("Refers to a single pass through the entire training dataset during the training of a model. In other words, an epoch is a measure of the number of times the model has seen the entire training data.")
    
    clip = st.number_input("Enter the clip value", value=0)
    with st.expander("More information"):
        st.write("The clip number is a hyperparameter that determines the maximum value that the gradient can take. If the gradient exceeds this value, it is clipped (i.e., truncated to the maximum value).")
    
    

    st.markdown("### Enter the FairGNN parameters")
    lr_fairgnn = st.number_input("Enter the learning rate for FairGNN")
    epochs_fairgnn = st.number_input("Enter the number of epochs for FairGNN", value=0)
    with st.expander("More information"):
        st.write("Refers to a single pass through the entire training dataset during the training of a model. In other words, an epoch is a measure of the number of times the model has seen the entire training data.")
    sens_number = st.number_input("Enter the sens number", value=0)
    
    label_number = st.number_input("Enter the label number", value=0)
    
    
    alpha = st.number_input("Enter alpha value", value=0)
    with st.expander("More information"):
        st.write("Refers to the regularization parameter that controls the amount of L2 regularization applied to the model's weights during the training process.")
     
    beta = st.number_input("Enter beta value", value=0)
    with st.expander("More information"):
        st.write("Refers to the momentum parameter that controls how much the optimizer should take into account the previous update when computing the current update to the model's weights during the training process.")
    

if "RHGN" in model_type and "CatGCN" in model_type:
    st.markdown("### Enter the general parameters")
    seed = st.number_input("Enter the prefered seed number", value=0)
    #predict_attr = st.text_input("Enter the prediction label")
    #sens_attr = st.text_input("Enter the senstive attribute")

    st.markdown("### Enter the RHGN parameters")
    num_hidden = st.text_input("Enter the number of hidden layers")
    lr_rhgn = st.number_input("Enter the learning rate", value=0)
    epochs_rhgn = st.number_input("Enter the number of epochs", value=0)
    clip = st.number_input("Enter the clip value", value=0)

    st.markdown("### Enter the CatGCN parameters")
    weight_decay = st.number_input("Enter the weight decay value" )
    lr_catgcn = st.number_input("Enter the learning rate", value=0)
    epochs_catgcn = st.number_input("Enter the number of epochs", value=0)
    diag_probe = st.number_input("Enter the diag probe value" , value=0)
    graph_refining = st.selectbox("Choose the graph refining approach", ("agc", "fignn", "none"))
    grn_units = st.number_input("Enter the grn units value" , value=0)
    bi_interaction = st.selectbox("Choose the bi-interaction approach", ("nfm", "none"))


elif "RHGN" in model_type and len(model_type) == 1:
    st.markdown("### Enter the general paramaters")
    seed = st.number_input("Enter the prefered seed number", value=0)
    #lr = st.number_input("Enter the learning rate", value=0)
    #epochs = st.number_input("Enter the number of epochs", value=0)
    #predict_attr = st.text_input("Enter the prediction label")
    #sens_attr = st.text_input("Enter the senstive attribute")


    st.markdown("### Enter the RHGN parametrs")
    num_hidden = st.number_input("Enter the number of hidden layers", value=0)
    lr_rhgn = st.number_input("Enter the learning rate")
    epochs_rhgn = st.number_input("Enter the number of epochs", value=0)
    clip = st.number_input("Enter the clip value" , value=0)

elif "FairGNN" in model_type and len(model_type) == 1:
    st.markdown("### Enter the general parameters")
    seed = st.number_input("Enter the prefered seed number" , value=0)
    #lr = st.number_input("Enter the learning rate" , value=0)
    #epochs = st.number_input("Enter the number of epochs" , value=0)
    #predict_attr = st.text_input("Enter the prediction label")
    #sens_attr = st.text_input("Enter the senstive attribute")


    st.markdown("### Enter the FairGNN parameters")
    #lr_fairgnn = st.number_input("Enter the learning rate" , value=0)
    epochs_fairgnn = st.number_input("Enter the number of epochs" , value=0)
    sens_number =  st.number_input("Enter the sens number" , value=0)
    label_number = st.number_input("Enter the label number", value=0)
    num_hidden = st.number_input("Enter the hidden layer number" , value=0)
    alpha = st.number_input("Enter alpha value" , value=0)
    beta = st.number_input("Enter beta value", value=0)


elif "CatGCN" in model_type and len(model_type) == 1:
    st.markdown("### Enter the general paramaters")
    seed = st.number_input("Enter the prefered seed number", value=0)
    #lr = st.number_input("Enter the learning rate" , value=0)
    #epochs = st.number_input("Enter the number of epochs" , value=0)
    #predict_attr = st.text_input("Enter the prediction label")
    #sens_attr = st.text_input("Enter the senstive attribute")

    st.markdown("### Enter the CatGCN parameters")
    weight_decay = st.number_input("Enter the weight decay value")
    lr_catgcn = st.number_input("Enter the learning rate")
    epochs_catgcn = st.number_input("Enter the number of epochs" , value=0)
    diag_probe = st.number_input("Enter the diag probe value" , value=0)
    graph_refining = st.multiselect("Choose the graph refining approach", ["agc", "fignn", "none"])
    grn_units = st.number_input("Enter the grn units value" , value=0)
    bi_interaction = st.multiselect("Choose the bi-interaction approach", ["nfm", "none"])



if len(model_type) != 0:
    if st.button("Begin experiment"):
        with st.spinner("Loading..."):

            time.sleep(2)
            ssh = paramiko.SSHClient()
            port = 443
            # Automatically add the server's host key (for the first connection only)
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            ssh.connect('https://dtdh206.cs.uni-magdeburg.de:443')
            #ssh.connect('141.44.31.206', port=443, banner_timeout=200)
            stdin, stdout, stderr =  ssh.exec_command('ls')
            print(stdout)

            if len(model_type) == 1 and 'FairGNN' in model_type:
                stdin, stdout, stderr = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 -W ignore main.py --seed {} --epoch {} --model GCN --sens_number {} --num_hidden {} --acc 0.20 --roc 0.20 --alpha {} --beta {} --dataset_name {} --dataset_path ../nba.csv --dataset_user_id_name user_id --model_type FairGNN --type 1 --sens_attr {} --predict_attr {} --label_number 100 --no-cuda True --special_case True --neptune_project mohamed9/FairGNN-Alibaba --neptune_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Nzc0MTIzMy0xMjRhLTQ0OGQtODE5Mi1mZjE3MDE0MGFhOGMifQ=='.format(seed, epochs_fairgnn, sens_number, num_hidden, alpha, beta, dataset, sens_attr, predict_attr))
            if len(model_type) == 1 and 'RHGN' in model_type:
                if predict_attr == 'final_gender_code':
                    predict_attr = 'bin_gender'
                if sens_attr == 'age_level':
                    sens_attr = 'bin_age'
                stdin, stdout, stderr = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 -W ignore main.py --seed {} --gpu 0 --dataset_path ../ --max_lr {} --num_hidden {} --clip {} --epochs {} --label {} --sens_attr {} --type 1 --model_type RHGN --dataset_name {} --dataset_user_id_name userid --special_case True --neptune_project mohamed9/FairGNN-Alibaba --neptune_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Nzc0MTIzMy0xMjRhLTQ0OGQtODE5Mi1mZjE3MDE0MGFhOGMifQ=='.format(seed, lr_rhgn, num_hidden, clip, epochs_rhgn, predict_attr, sens_attr, dataset))
            # CatGCN
            if len(model_type) == 1 and 'CatGCN' in model_type:
                if predict_attr == 'final_gender_code':
                    predict_attr = 'bin_gender'
                if sens_attr == 'age_level':
                    sens_attr = 'bin_age'
                stdin, stdout, stderr = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 -W ignore main.py --seed {} --gpu 0 --lr {} --weight_decay {} --dropout 0.1 --diag-probe {} --graph-refining {} --aggr-pooling mean --grn_units {} --bi-interaction {} --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none --aggr-style sum --balance-ratio 0.7 --sens_attr {} --label {} --dataset_name {} --dataset_path ../ --type 1 --model_type CatGCN --dataset_user_id_name userid --alpha 0.5 --special_case True --neptune_project mohamed9/FairGNN-Alibaba --neptune_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Nzc0MTIzMy0xMjRhLTQ0OGQtODE5Mi1mZjE3MDE0MGFhOGMifQ=='.format(seed, lr_catgcn, weight_decay, diag_probe, graph_refining, grn_units, bi_interaction, sens_attr, predict_attr, dataset))

            # FairGNN and RHGN
            if len(model_type) == 2 and 'FairGNN' in model_type and 'RHGN' in model_type:
                if predict_attr == 'final_gender_code':
                    label = 'bin_gender'
                if sens_attr == 'age_level':
                    sens_attr_rhgn = 'bin_age'
                stdin, stdout, stderr = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 -W ignore main.py --seed {} --epochs {} --model GCN --sens_number {} --num_hidden {} --acc 0.20 --roc 0.20 --alpha {} --beta {} --dataset_name {} --dataset_path ../nba.csv --dataset_user_id_name user_id --model_type FairGNN RHGN --type 1 --sens_attr {} --label {} --predict_attr {} --label_number 100 --no-cuda True --max_lr {} --clip {} --epochs_rhgn {}  --special_case True --neptune_project mohamed9/FairGNN-Alibaba --neptune_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Nzc0MTIzMy0xMjRhLTQ0OGQtODE5Mi1mZjE3MDE0MGFhOGMifQ=='.format(seed, epochs_fairgnn, sens_number, num_hidden, alpha, beta, dataset, sens_attr, predict_attr, predict_attr, lr_rhgn, clip, epochs_rhgn))
                print('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && /home/abdelrazek/anaconda3/envs/test/bin/python3 -W ignore main.py --seed {} --epochs {} --model GCN --sens_number {} --num_hidden {} --acc 0.20 --roc 0.20 --alpha {} --beta {} --dataset_name {} --dataset_path ../nba.csv --dataset_user_id_name user_id --model_type FairGNN RHGN --type 1 --sens_attr {} --label {} --predict_attr {} --label_number 100 --no-cuda True --max_lr {} --clip {} --epochs_rhgn {}  --special_case True --neptune_project mohamed9/FairGNN-Alibaba --neptune_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Nzc0MTIzMy0xMjRhLTQ0OGQtODE5Mi1mZjE3MDE0MGFhOGMifQ=='.format(seed, epochs_fairgnn, sens_number, num_hidden, alpha, beta, dataset, sens_attr, predict_attr, predict_attr, lr_rhgn, clip, epochs_rhgn))
                #stdin, stdout, stderr = ssh.exec_command('cd /home/abdelrazek/framework-for-fairness-analysis-and-mitigation-main && ls')
                
            output_queue = queue.Queue()
            output_thred = threading.Thread(target=read_output, args=(stderr, output_queue))
            output_thred.start()
            
            while True:
                try:
                    line = output_queue.get_nowait()
                    #st.text(line)
                except queue.Empty:
                    if output_thred.is_alive():
                        continue
                    else:
                        break

            output_thred.join()
            all_output = []
            for line in stdout:
                print(line.strip())
                #st.text(line.strip())
                if "Test_final:" in line and 'FairGNN' in model_type:
                    result = line.strip()
                    #st.text(result)
                if 'accuracy' in line and 'RHGN' in model_type:
                    #st.text(line.strip())
                    line = line.strip() + 'end'
                    acc = re.search('accuracy                         (.+?)end', line)
                    acc = acc.group(1)
                    acc_rhgn = acc.split()[0]
                if 'F1 score:' in line:
                    f1 = '.'.join(line.split('.')[0:2])
                    f1_rhgn = '{:.3f}'.format(float(f1.split()[-1]))
                if 'Statistical Parity Difference (SPD):' in line:
                    spd_rhgn = '{:.3f}'.format(float(line.split()[-1]))

                if 'Equal Opportunity Difference (EOD):' in line:
                    eod_rhgn = '{:.3f}'.format(float(line.split()[-1]))

                if 'Overall Accuracy Equality Difference (OAED):' in line:
                    oaed_rhgn = '{:.3f}'.format(float(line.split()[-1]))

                if 'Treatment Equality Difference (TED):' in line:
                    ted_rhgn = '{:.3f}'.format(float(line.split()[-1]))
                #all_output.append(line.strip())
            # Close the connection
            ssh.close()

        st.success("Done!")

        
        st.markdown("## Training Results:")
        print(len(model_type))
        print(model_type)
        if len(model_type) == 1 and 'FairGNN' in model_type:
            st.text(result)
            acc = re.search('accuracy:(.+?)roc', result)
            f1 = re.search('F1:(.+?)acc_sens', result)

            spd = re.search('parity:(.+?)equality', result)
            eod = re.search('equality:(.+?)oaed', result)
            oaed = re.search('oaed:(.+?)treatment equality', result)
            ted = re.search('treatment equality(.+?)end', result)
            data = {'Model': [model_type],
                'Accuracy': [acc.group(1)],
                'F1': [f1.group(1)],
                'SPD': [spd.group(1)],
                'EOD': [eod.group(1)],
                'OAED': [oaed.group(1)],
                'TED': [ted.group(1)]
                }
            
        elif len(model_type) == 1 and 'RHGN' in model_type:
            #print('all_output:', all_output)
            data =  {'Model': [model_type],
            'Accuracy': [acc_rhgn],
            'F1': [f1_rhgn],
            'SPD': [spd_rhgn],
            'EOD': [eod_rhgn],
            'OAED': [oaed_rhgn],
            'TED': [ted_rhgn]
            }

        elif len(model_type) == 2 and 'RHGN' in model_type and 'FairGNN' in model_type:

            acc = re.search('a:(.+?)roc', result)
            f1 = re.search('F1:(.+?)acc_sens', result)

            spd = re.search('parity:(.+?)equality', result)
            eod = re.search('equality:(.+?)oaed', result)
            oaed = re.search('oaed:(.+?)treatment equality', result)
            ted = re.search('treatment equality(.+?)end', result)

            ind_fairgnn = model_type.index('FairGNN')
            ind_rhgn = model_type.index('RHGN')
            data =  {'Model': [model_type[ind_fairgnn], model_type[ind_rhgn]],
            'Prediction label': [predict_attr, predict_attr],
            'Sensitive attribute': [sens_attr, sens_attr],
            'Accuracy': [acc.group(1), acc_rhgn],
            'F1': [f1.group(1), f1_rhgn],
            'SPD': [spd.group(1), spd_rhgn],
            'EOD': [eod.group(1), eod_rhgn],
            'OAED': [oaed.group(1), oaed_rhgn],
            'TED': [ted.group(1), ted_rhgn]
            }

        df = pd.DataFrame(data)

        #st.dataframe(df, width=5000)
        # set the display options for the DataFrame
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)

        

        # display the DataFrame in Streamlit
        st.write(df)

    #st.write("The logs of the experiment can be found at: mohamed9/Experiments-RHGN-CatGCN-Alibaba")
    #st.markdown("The logs of the experiment can be found at: **mohamed9/Experiments-RHGN-FairGNN-Alibaba**")

    

