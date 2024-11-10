python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molbace2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molbbbp2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molclintox2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molesol2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molfreesolv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molhiv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > mollipo2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molmuv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > molsider2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > moltox212.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling add --svm True > moltoxcast2.txt




python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molbace4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molbbbp4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molclintox4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molesol4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molfreesolv4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molhiv4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > mollipo4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molmuv4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > molsider4.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > moltox214.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling mean --svm True > moltoxcast4.txt


python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molbace6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molbbbp6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molclintox6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molesol6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molfreesolv6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molhiv6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > mollipo6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molmuv6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > molsider6.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > moltox216.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pretrained_path weight/s2gaesvm-GCN12000_zinc12k_dm_2_hidd128-0.8-3-256_model.pth --pooling max --svm True > moltoxcast6.txt
