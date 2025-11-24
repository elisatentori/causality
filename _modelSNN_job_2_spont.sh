#!/bin/bash

#SBATCH --job-name=spont
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1%1 
#SBATCH --output=/dev/null #ciao.out #
#SBATCH --mem=30G
#SBATCH -t 1:00:00
#SBATCH --partition=brains
# #SBATCH --nodelist=brain01

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- avoid thread oversubscription ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1           # numba single-thread
export NUMBA_THREADING_LAYER=workqueue
export MPLBACKEND=Agg
[ -n "${SLURM_TMPDIR:-}" ] && export JOBLIB_TEMP_FOLDER="$SLURM_TMPDIR"


array_config="./_modelSNN_config_sim_2.txt"

mkdir -p _logs/
mkdir -p _logs/_2_logs_jobs_spontaneous

# -----------------------------
# Fixed params (edit as needed)
# -----------------------------
path_results='./Data_SNNmodel/'

#modules='1'
#local_II=1                             # 0 = noII, 1 = II
tau_R=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $3}' "$array_config")

dist_rule='EDR'
nNeurons=1000
tau_AMPA=3.
beta_E=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $2}' "$array_config")
beta_I=0.8
gain_AMPA=1.0
gain_GABA=1.

time_sec=1800.
g_E=2.0
g_I=1.

I_intensity_exc=13.5
I_intensity_inh=10.
rate_exc=1.2
rate_inh=1.

#for local_II in 0 1; do
for modules in '2' '1'; do

    local_II=0
    # -----------------------------
    # Conditional params
    # -----------------------------
    # tau_GABA depends on local_II
    if [ "$local_II" -eq 0 ]; then
      tau_GABA=50.
      conn_tag='noII'
    else
      tau_GABA=10.
      conn_tag='II'
    fi
    
    #out_name="network${nNeurons}_${modules}mod_${dist_rule}_${conn_tag}__tauR_${tau_R}__g_I_${g_I}__betaE_${beta_E}__gain_AMPA_${gain_AMPA}"
    out_name="network${nNeurons}_${modules}mod_${dist_rule}_${conn_tag}__tauR_${tau_R}__g_E_${g_E}__betaE_${beta_E}__gain_AMPA_${gain_AMPA}"
    echo $out_name
    
    sleep $((RANDOM % 4 + 1))
    
    # -----------------------------
    # Run spontaneous activity
    # -----------------------------
    python3.9 -u _modelSNN_2_sim_spontaneous.py $path_results $modules $dist_rule $nNeurons $local_II $tau_AMPA $tau_GABA $tau_R $beta_E $beta_I $gain_AMPA $gain_GABA $time_sec $g_E $g_I $I_intensity_exc $I_intensity_inh $rate_exc $rate_inh $out_name > "_logs/_2_logs_jobs_spontaneous/output_"${out_name}".log" 2> "_logs/_2_logs_jobs_spontaneous/error_"${out_name}".log"
    
done
#done    
