infea=v
tarfea=v
mepoch=200
bs=64
lr=0.001
patience=5
diff_alpha=0.8
k_hop=1
latent_dim=64
model=DEFP4D

data_path='./data/METR-LA/raw_data.pkl'
python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler None --stop_based 'train_total' --seed 42 \
                --max_epoch $mepoch --lr $lr --batch_size $bs \
                --diff_alpha $diff_alpha --latent_dim $latent_dim --k_hop $k_hop\
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --data_path $data_path --unknown_nodes_path './data/METR-LA/unknown_nodes.npy' \
                --fp_step 300 --sp_adj --no_scale --dloader_name METR-LA
                
                   



data_path='./data/PEMS-BAY/raw_data.pkl'

python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler None --stop_based 'train_total' --seed 20 \
                --max_epoch $mepoch --lr $lr --batch_size $bs \
                --diff_alpha $diff_alpha --latent_dim $latent_dim --k_hop $k_hop\
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --data_path $data_path --unknown_nodes_path './data/PEMSBAY/unknown_nodes.npy' \
                --fp_step 300 --sp_adj --no_scale --dloader_name PEMSBAY


data_path='./data/PEMSD7M/raw_data.pkl'
python main.py  --model $model \
                --loss mse --patience $patience --num_workers 8  --scheduler None --stop_based 'train_total' --seed 30 \
                --max_epoch $mepoch --lr $lr --batch_size $bs \
                --diff_alpha $diff_alpha --latent_dim $latent_dim --k_hop $k_hop\
                --t_mark --miss_mark \
                --input_features $infea --target_features $tarfea \
                --input_len 12 --pred_len 0 --look_back 12 --slide_step 12 \
                --data_path $data_path --unknown_nodes_path './data/PEMSD7M/unknown_nodes.npy' \
                --fp_step 300 --sp_adj --no_scale --dloader_name PEMSD7M
