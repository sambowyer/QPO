partition=cnu
code=$PROJ_CODE

gpu_type="A100"
num_gpus=1
mem=124

NUM_CPU=16
TIME=48

# algos=("QPO" "grpo" "dr_grpo" "dapo" "optimal")
# algos=("QPO")
algos=("grpo")
# datasets=("countdown" "gsm8k")
datasets=("gsm8k")

Q_betas=(0.1 1.0 10.0)
# Q_betas=(0.01 0.1 1.0 10.0)
# Q_betas=(2.0)

lrs=(5e-7 1e-6 5e-6)
# lrs=(5e-7 5e-6)
# lrs=(1e-6)

# split_rewards=(True False)
split_rewards=(False)
# beta_format=(0.01 0.1 1.0)
# beta_format=(1.0 2.0 5.0 10.0)
# beta_format=(20.0)

combine_rewards=True


for algo in ${algos[@]}; do
    for dataset in ${datasets[@]}; do
        for lr in ${lrs[@]}; do
            if [ $algo == "QPO" ]; then
                for Q_beta in ${Q_betas[@]}; do
                    for split_reward in ${split_rewards[@]}; do
                        if [ $split_reward == "True" ]; then
                            for beta_format in ${beta_format[@]}; do
                                lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr-Qbeta$Q_beta-split_reward-beta_f$beta_format-AC0 --conda-env grpo \
                                    --cmd "python train.py --algo $algo --dataset $dataset --Q_beta $Q_beta --learning_rate $lr --split_rewards --beta_format $beta_format --Q_A 0.0 --Q_c 0.0"
                            done
                        else
                            lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr-Qbeta$Q_beta --conda-env grpo \
                                --cmd "python train.py --algo $algo --dataset $dataset --Q_beta $Q_beta --learning_rate $lr --EOS_format_reward"
                        fi
                    done
                done
            else
                if [ $combine_rewards == "True" ]; then
                    lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr-combine_rewards --conda-env grpo \
                        --cmd "python train.py --algo $algo --dataset $dataset --learning_rate $lr --combine_rewards"
                else
                    lbatch -c $NUM_CPU -g $num_gpus --gputype $gpu_type -m $mem -t $TIME -a $code -q $partition -n $algo-$dataset-$lr --conda-env grpo \
                        --cmd "python train.py --algo $algo --dataset $dataset --learning_rate $lr"
                fi
            fi
        done
    done
done
