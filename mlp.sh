python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pooling add --freeze_backbone > mlpbace1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pooling add --freeze_backbone > mlpbbbp1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pooling add --freeze_backbone > mlpclintox1.txt 
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pooling add --freeze_backbone > mlpesol1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pooling add --freeze_backbone > mlpfreesolv1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pooling add --freeze_backbone > mlphiv1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pooling add --freeze_backbone > mlplipo1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pooling add --freeze_backbone > mlpmuv1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pooling add --freeze_backbone > mlpsider1.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pooling add --freeze_backbone > mlptox211.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pooling add --freeze_backbone > mlptoxcast1.txt

python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pooling add > mlpbace2.txt 
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pooling add > mlpbbbp2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pooling add > mlpclintox2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pooling add > mlpesol2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pooling add > mlpfreesolv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pooling add > mlphiv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pooling add > mlplipo2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pooling add > mlpmuv2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pooling add > mlpsider2.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pooling add > mlptox212.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pooling add > mlptoxcast2.txt

python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbace.json --pooling add --scratch    > mlpbace3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molbbbp.json --pooling add --scratch   > mlpbbbp3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molclintox.json --pooling add --scratch   > mlpclintox3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molesol.json --pooling add --scratch   > mlpesol3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molfreesolv.json --pooling add --scratch   > mlpfreesolv3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molhiv.json --pooling add --scratch   > mlphiv3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-mollipo.json --pooling add --scratch   > mlplipo3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molmuv.json --pooling add --scratch   > mlpmuv3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-molsider.json --pooling add --scratch   > mlpsider3.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltox21.json --pooling add --scratch   > mlptox213.txt
python3 s2gae_transfer.py --dataset_config ./molecules_config/ogbg-moltoxcast.json --pooling add --scratch   > mlptoxcast3.txt
