# Interventional Connectivity

Code supporting <a href="https://www.biorxiv.org/content/10.1101/2025.04.29.651327v2" target="_blank">"Spontaneous Dynamics Predict the Effects of Targeted Intervention in Hippocampal Neuronal Cultures" </a>, bioRxiv

## Data shared

<ul>
  <li> <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1/Data" target="_blank"> Data_MaxOne/Culture1REC1/Data </a>: 30-minutes spontaneous activity of cultured neurons from the hippocampus of rat embryo (DIV 20-30) plated on <a href="https://www.mxwbio.com/products/maxone-mea-system-microelectrode-array/" target="_blank"> MaxOne Single-Well HD-MEA System </a> (sample 1) </li>
  <li> <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture2REC1/Data" target="_blank"> Data_MaxOne/Culture2REC1/Data </a>: 30-minutes spontaneous activity of cultured neurons from rat embryio hippocampus (sample 2) </li>
</ul>

## Main scripts

<a href="https://github.com/elisatentori/causality/blob/main/results1.ipynb" target="_blank"> results1</a>: reproduce the main results shared in the manuscripts. 
<ul>
<li>Contstruct and characterize the perturbome (via Interventional Connectivity).</li>
<li>Apply the developed Effective Connectivity (EC) framework to <br>
  –  evaluate the predictive power of EC; <br>
  –  assess EC validity as a proxy of causal influence.
</li>
</ul>

## Dependencies

<a href="https://github.com/elisatentori/EC_calculation" target="_blank"> EC_calculation</a>: computing Effective Connectivity metrics (Delayed Transfer Entropy, Signed-Cross Correlation, Cross-Covariance) from spike-trains. Performing significance jittering test.

## utils

<ul>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/load_data.py" target="_blank"> load_data.py</a>: load HD-MEA channels map and spike-trains from Matlab structure (<a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1/Data" target="_blank"> Data_MaxOne/Culture1REC1/Data </a>) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/spikeDataProcessor.py" target="_blank"> spikeDataProcessor.py</a>: class to manage recorded spike-trains from both spontaneous and evoked activity.</li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/interventional.py" target="_blank"> interventional.py</a>: compute Interventional Connectivity between stimulating-recording channels pairs (see our paper for details) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/load_EC.py" target="_blank"> load_EC.py</a>: load EC results, previously computed via <a href="https://github.com/elisatentori/EC_calculation" target="_blank"> EC_calculation</a> </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/distance.py" target="_blank"> distance.py</a>: correct EC and IC metrics for spatial dependence. </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/network.py" target="_blank"> network.py</a>: compute the shortest-paths of effective networks using Dijkstra's algorithm (networkx package) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/plot.py" target="_blank"> plot.py</a>: library to visualize results</li>
  
</ul>

## Results

Stored in <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1" target="_blank"> Data_MaxOne/Culture1REC1 </a> and <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture2REC1" target="_blank"> Data_MaxOne/Culture2REC1 </a> subfolders.

## python dependencies

numpy, matplotlib, scipy, sklearn, statsmodels, joblib, networkx, seaborn
