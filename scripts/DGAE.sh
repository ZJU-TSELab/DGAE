infea=v
tarfea=v
mepoch=200
bs=64
lr=0.005
diff_alpha=0.8
k_hop=1
latent_dim=128
scheduler_patience=5
model=DGAE

scheduler='step'
patience=5
data_path='./data/METR-LA/raw_data.pkl'
python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler $scheduler --stop_based 'train_total' --seed 42 \
                --max_epoch $mepoch --lr $lr --batch_size $bs --scheduler_patience $scheduler_patience\
                --diff_alpha $diff_alpha --sp_adj --latent_dim $latent_dim --k_hop $k_hop\
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --dloader_name METRLA --data_path $data_path --unknown_nodes_path './data/METR-LA/unknown_nodes.npy' \
                --fp_step 4 



scheduler='None'
patience=10
data_path='./data/PEMSBAY/raw_data.pkl'

python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler $scheduler --stop_based 'train_total' --seed 20 \
                --max_epoch $mepoch --lr $lr --batch_size $bs --scheduler_patience $scheduler_patience\
                --diff_alpha $diff_alpha --latent_dim $latent_dim --k_hop $k_hop\
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --dloader_name PEMSBAY --data_path $data_path --unknown_nodes_path './data/PEMSBAY/unknown_nodes.npy' \
                --fp_step 4 


data_path='./data/PEMSD7M/raw_data.pkl'
scheduler='None'
patience=10
python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler $scheduler --stop_based 'train_total' --seed 30 \
                --max_epoch $mepoch --lr $lr --sp_adj --batch_size $bs --scheduler_patience $scheduler_patience \
                --diff_alpha $diff_alpha --latent_dim $latent_dim --k_hop $k_hop \
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --dloader_name PEMSD7M --data_path $data_path --unknown_nodes_path './data/PEMSD7M/unknown_nodes.npy' \
                --fp_step 4 