#!/bin/bash

#SBATCH --job-name=SP
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --array=[1-3,8-10]%3
#SBATCH --output=/dev/null
#SBATCH --mem=20G
#SBATCH -t 24:00:00

cd $SLURM_SUBMIT_DIR


array_config='./config_EC.txt' # file with the configurations you want to compute

mkdir _logs_jobs_computeSP_

numCulture=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $2}' $array_config)

main_path='/home/tentori/IC_EC_package/Data_MaxOne/'
folder_name=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $5}' $array_config)
meas=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $6}' $array_config)
type_meas=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $7}' $array_config)
DeltaT=30
bs=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $3}' $array_config)

out_name=$folder_name'_'${meas}'_'${type_meas}
echo $out_name

sleep $((RANDOM % 4 + 1))

alpha_th=0.001

python3.6 -u compute_SP.py $main_path $folder_name $meas $type_meas $DeltaT $bs $alpha_th  > "./_logs_jobs_computeSP_/"${out_name}"_output_"${bs}".log" 2> "./_logs_jobs_computeSP_/error_"${out_name}"_"${bs}".log"


#load_KS=0
#python3.6 -u code/compute_KS_parameters.py $main_path $folder_name $Ntrials $meas $type_meas $tstart $DeltaT $Del_max $bs $load_KS > "./logs_jobs_KSopt/del"${Del_max}"_"${out_name}"_output_"${bs}".log" 2> "./logs_jobs_KSopt/error_del"${Del_max}"_"${out_name}"_"${bs}".log"

### to launch the job:  
### >> sbatch computeSP_job.sh 