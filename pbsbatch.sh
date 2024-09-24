#!/bin/sh
### Set the job name (for your reference)
#PBS -N s2gae_gc
### Set the project name, your department code by default
#PBS -P col870.course
### Request email when job begins and ends, don't change anything on the below line 
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=1:ngpus=1:centos=icelake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=24:00:00
#PBS -l software=PYTHON

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module () {
        eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load apps/anaconda/3
source activate ~/miniconda3/envs/s2gae
module unload apps/anaconda/3

# python s2gae_gc_acc.py --dataset MUTAG --dropout 0 --batch_size 2048 --num_layers 3 --decode_layers 4 >> gc_exp.sh
# python s2gae_gc_acc.py --dataset PROTEINS --mask_ratio 0.6 >> gc_exp.sh
python s2gae_gc_acc.py --dataset NCI1 --pooling max >> gc_exp.sh
python s2gae_gc_acc.py --dataset IMDB-MULTI --pooling max >> gc_exp.sh
python s2gae_gc_acc.py --dataset IMDB-BINARY >> gc_exp.sh
python s2gae_gc_acc.py --dataset REDDIT-BINARY >> gc_exp.sh
python s2gae_gc_acc.py --dataset COLLAB-BINARY >> gc_exp.sh
