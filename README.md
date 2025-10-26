# Interventional Connectivity

Code supporting <a href="https://www.biorxiv.org/content/10.1101/2025.04.29.651327v2" target="_blank">"Spontaneous Dynamics Predict the Effects of Targeted Intervention in Hippocampal Neuronal Cultures" </a>, bioRxiv

## Data shared

30-minutes spontaneous activity of cultured neurons from the hippocampus of rat embryo (DIV 20-30) plated on <a href="https://www.mxwbio.com/products/maxone-mea-system-microelectrode-array/" target="_blank"> MaxOne Single-Well HD-MEA System.

<ul>
  <li> <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1/Data" target="_blank"> Data_MaxOne/Culture1REC1/Data </a>: sample 1  </li>
  <li> <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture2REC1/Data" target="_blank"> Data_MaxOne/Culture2REC1/Data </a>: sample 2  </li>
</ul>

## Main scripts

<ul>
  <li>
    Notebooks <a href="https://github.com/elisatentori/causality/blob/main/results1.ipynb" target="_blank"> results1</a> and <a href="https://github.com/elisatentori/causality/blob/main/results2.ipynb" target="_blank"> results2</a>: reproduce the main results shared in the manuscripts. 
    <ul>
      <li>Contstruct and characterize the perturbome (via Interventional Connectivity).</li>
      <li>Apply the developed Effective Connectivity (EC) framework to 
        <ul>
          <li> evaluate the predictive power of EC; </li>
          <li>  assess EC validity as a proxy of causal influence.</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>

## Dependency for EC calculation
<ul>
<li><a href="https://github.com/elisatentori/EC_calculation" target="_blank"> EC_calculation</a>: <br>
  computes Effective Connectivity metrics (Delayed Transfer Entropy, Signed-Cross Correlation, Cross-Covariance) from spike-trains, performing significance jittering test. </li>
</ul>

## utils

<ul>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/load_data.py" target="_blank"> utils/load_data.py</a>: load HD-MEA channels map and spike-trains from Matlab structure (<a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1/Data" target="_blank"> Data_MaxOne/Culture1REC1/Data </a>) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/spikeDataProcessor.py" target="_blank"> utils/spikeDataProcessor.py</a>: class to manage recorded spike-trains from both spontaneous and evoked activity.</li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/interventional.py" target="_blank"> utils/interventional.py</a>: compute Interventional Connectivity between stimulating-recording channels pairs (see our paper for details) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/load_EC.py" target="_blank"> utils/load_EC.py</a>: load EC results, previously computed via <a href="https://github.com/elisatentori/EC_calculation" target="_blank"> EC_calculation</a> </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/distance.py" target="_blank"> utils/distance.py</a>: correct EC and IC metrics for spatial dependence. </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/network.py" target="_blank"> utils/network.py</a>: compute the shortest-paths of effective networks using Dijkstra's algorithm (networkx package) </li>
  <li> <a href="https://github.com/elisatentori/causality/blob/main/utils/plot.py" target="_blank"> utils/plot.py</a>: library to visualize results</li>
</ul>

## Results

<ul>
  <li> Data and results from EC/IC computation are stored in <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture1REC1" target="_blank"> Data_MaxOne/Culture1REC1 </a> and <a href="https://github.com/elisatentori/causality/tree/main/Data_MaxOne/Culture2REC1" target="_blank"> Data_MaxOne/Culture2REC1 </a> subfolders.
  </li>
  <li> Analyses and plots are stored in <a href="https://github.com/elisatentori/causality/tree/main/imgs" target="_blank"> imgs </a> and <a href="https://github.com/elisatentori/causality/tree/main/imgs2  target="_blank"> imgs2 </a> subfolders.
  </li>
</ul>

## python dependencies

numpy, matplotlib, scipy, sklearn, statsmodels, joblib, networkx, seaborn



<br><br><br>
All rights reserved. Copyright (c) 2025, University of Padua, Italy <br>
Author: Elisa Tentori. LiPh Lab - NeuroChip Lab, University of Padua, Italy

If you use the package, please cite <a href="https://www.biorxiv.org/content/10.1101/2025.04.29.651327v2" target="_blank">"Spontaneous Dynamics Predict the Effects of Targeted Intervention in Hippocampal Neuronal Cultures" </a>, bioRxiv
