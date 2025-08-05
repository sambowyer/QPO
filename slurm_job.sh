partition=cnu
code=$PROJ_CODE

gpu_type="A100"
num_gpus=1
mem=124

NUM_CPU=16
TIME=48

# algos=("QPO" "grpo" "dr_grpo" "dapo" "optimal")
algos=("QPO")
# datasets=("countdown" "gsm8k")
datasets=("gsm8k")

Q_betas=(0.01 0.1 1.0)
# lrs=(5e-7 1e-6 5e-6)
lrs=(1e-6)

for algo in ${algos[@]}; do
    for dataset in ${datasets[@]}; do
        for lr in ${lrs[@]}; do
            if [ $algo == "QPO" ]; then
                for Q_beta in ${Q_betas[@]}; do
                    lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr-Qbeta$Q_beta --conda-env grpo \
                        --cmd "python train.py --algo $algo --dataset $dataset --Q_beta $Q_beta --learning_rate $lr"
                done
            else
                lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr --conda-env grpo \
                    --cmd "python train.py --algo $algo --dataset $dataset --learning_rate $lr"
            fi
        done
    done
done
