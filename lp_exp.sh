python s2gae_small_lp.py --dataset "Cora" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_small_lp.py --dataset "CiteSeer" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_small_lp.py --dataset "PubMed" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_small_lp.py --dataset "BlogCatalog" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_small_lp.py --dataset "Photo" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt

python s2gae_large_lp.py --dataset "ogbl-ddi" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_large_lp.py --dataset "ogbl-collab" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
python s2gae_large_lp.py --dataset "ogbl-ppa" --runs 10 --use_sage "SAGE" >> logs/lp_sage.txt
# python s2gae_nc_auc.py --dataset "proteins" --runs 10 >> logs/nc_sage.txt