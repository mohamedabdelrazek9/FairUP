python train_fairGNN.py \
        --seed=42 \
        --epochs=1800 \
        --model=GAT \
        --sens_number=200 \
        --dataset=pokec_n \
        --num-hidden=64 \
        --attn-drop=0.0 \
        --acc=0.688 \
        --roc=0.745 \
        --alpha=4 \
        --beta=0.01