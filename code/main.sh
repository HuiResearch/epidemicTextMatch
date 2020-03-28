export CUDA_VISIBLE_DEVICES=0

python data_aug.py

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--model_name_or_path ../user_data/model_data/roberta_wwm_large \
--data_dir ../user_data/tmp_data/Kfold/$i \
--output_dir ../user_data/tmp_data/checkpoints/roberta_wwm_large/$i \
--learning_rate 2e-5

done

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--model_name_or_path ../user_data/model_data/roberta_pair \
--data_dir ../user_data/tmp_data/Kfold/$i \
--output_dir ../user_data/tmp_data/checkpoints/roberta_pair/$i \
--learning_rate 5e-6

done

for ((i=0;i<5;i++));
do

python train.py \
--do_train \
--do_eval_during_train \
--model_name_or_path ../user_data/model_data/ernie \
--data_dir ../user_data/tmp_data/Kfold/$i \
--output_dir ../user_data/tmp_data/checkpoints/ernie/$i \
--learning_rate 4e-5

done

python predict.py \
--vote_model_paths ../user_data/tmp_data/checkpoints/roberta_wwm_large,../user_data/tmp_data/checkpoints/roberta_pair,../user_data/tmp_data/checkpoints/ernie \
--predict_file ../data/Dataset/test.csv \
--predict_result_file ../prediction_result/result.csv



