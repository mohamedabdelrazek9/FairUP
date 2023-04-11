import streamlit as st
import pandas as pd
import os
import paramiko
import warnings
import re
import time


def FairGNN_NBA():
    #dataset = st.selectbox("Dataset", ("NBA"))
    dataset = st.text_input('Dataset', 'NBA', disabled=True)
    dataset = 'nba'
    predict_attr = st.text_input("Prediction label", 'SALARY', disabled=True)
    sens_attr = st.text_input("Sensitive attribute", 'country', disabled=True)


    model_type = st.text_input("Models to train", 'FairGNN', disabled=True)

    st.markdown("### General parameters")
    seed = st.number_input("Prefered seed number" , value=42, disabled=True)

    st.markdown("### FairGNN parameters")
    lr_fairgnn = st.number_input("Learning rate", value=0.01, disabled=True)
    epochs_fairgnn = st.number_input("Number of epochs" , value=2000, disabled=True)
    with st.expander("More information"):
        st.write("Refers to a single pass through the entire training dataset during the training of a model. In other words, an epoch is a measure of the number of times the model has seen the entire training data.")
    sens_number =  st.number_input("Sens number" , value=50, disabled=True)
    label_number = st.number_input("Label number", value=1000, disabled=True)
    num_hidden = st.number_input("Hidden layer number" , value=128, disabled=True)
    with st.expander("More information"):
        st.write("The number of hidden layers refers to the number of layers between the input layer and the output layer of a model.")
    alpha = st.number_input("Alpha value" , value=10, disabled=True)
    with st.expander("More information"):
        st.write("Refers to the regularization parameter that controls the amount of L2 regularization applied to the model's weights during the training process.")
    beta = st.number_input("Beta value", value=1, disabled=True)
    with st.expander("More information"):
        st.write("Refers to the momentum parameter that controls how much the optimizer should take into account the previous update when computing the current update to the model's weights during the training process.")

    return model_type, predict_attr, sens_attr

def RHGN_Alibaba():
    dataset = st.text_input('Dataset', 'Alibaba', disabled=True)
    dataset = 'alibaba'
    predict_attr = st.text_input('Prediction label', 'final_gender_code', disabled=True)
    sens_attr = st.text_input('Sensitive attribute', 'age_level', disabled=True)

    model_type = st.text_input("Models to train", 'RHGN', disabled=True)

    st.markdown("### General parameters")
    seed = st.number_input("Prefered seed number" , value=3, disabled=True)

    st.markdown("### RHGN parametrs")
    num_hidden = st.number_input("Hidden layer number", value=32, disabled=True)
    with st.expander("More information"):
        st.write("The number of hidden layers refers to the number of layers between the input layer and the output layer of a model.")
    lr_rhgn = st.number_input("Learning rate", value=0.1, disabled=True)
    with st.expander("More information"):
        st.write("Is a hyperparameter that controls the step size of the updates made to the weights during training. In other words, it determines how quickly the model learns from the data.")
    
    epochs_rhgn = st.number_input("Epochs", value=100, disabled=True)
    with st.expander("More information"):
        st.write("Refers to a single pass through the entire training dataset during the training of a model. In other words, an epoch is a measure of the number of times the model has seen the entire training data.")
    
    clip = st.number_input("Clip value", value=2, disabled=True)
    with st.expander("More information"):
        st.write("The clip number is a hyperparameter that determines the maximum value that the gradient can take. If the gradient exceeds this value, it is clipped (i.e., truncated to the maximum value).")

    return model_type, predict_attr, sens_attr

def CatGCN_Alibaba():
    dataset = st.text_input('Dataset', 'Alibaba', disabled=True)
    dataset = 'alibaba'
    predict_attr = st.text_input('Prediction label', 'final_gender_code', disabled=True)
    sens_attr = st.text_input('Sensitive attribute', 'age_level', disabled=True)

    model_type = st.text_input('Models to train', 'CatGCN', disabled=True)

    st.markdown("### General parameters")
    seed = st.number_input("Prefered seed number" , value=11, disabled=True)

    st.markdown("### CatGCN parameters")
    weight_decay = st.number_input("Weight decay value", value=0.01, disabled=True)
    with st.expander("More information"):
        st.write("The parameters that controls the amount the weights will exponentially decay to zero.")
    lr_catgcn = st.number_input("Learning rate", value=0.1, disabled=True)
    with st.expander("More information"):
        st.write("Is a hyperparameter that controls the step size of the updates made to the weights during training. In other words, it determines how quickly the model learns from the data.")
    epochs_catgcn = st.number_input("Number of epochs" , value=100, disabled=True)
    with st.expander("More information"):
        st.write("Refers to a single pass through the entire training dataset during the training of a model. In other words, an epoch is a measure of the number of times the model has seen the entire training data.")
    diag_probe = st.number_input("Diag probe value" , value=39, disabled=True)
    graph_refining = st.text_input("Graph refining approach", "agc", disabled=True)
    grn_units = st.number_input("Enter the grn units value" , value=64, disabled=True)
    bi_interaction = st.text_input("Bi-interaction approach", "nfm", disabled=True)

    return model_type, predict_attr, sens_attr

def experiment_begin(model_type, predict_attr, sens_attr):
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
