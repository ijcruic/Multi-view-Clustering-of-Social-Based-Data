# Multi-view-Clustering
### Python implementations of multi-view clustering on social-based data.


This repository contains python implementations of algorihtms used in multi-view clustering
of social-based data. The current algorithms implemented are as follows:

1. **Cross View Influence Clustering (CVIC)**: An algorithm which works with intermediate and late paradigm
clustering information
	- requires cluster labels and graphs for each view of the data
	- outputs the cluster labels for the objects and can also output cluster labels for clusters from
	the original view clusterings
	- suitbale for mid-sized data sets (up to ~10,000 objects)
	- references:
		- Cruickshank I., Carley K. M. *Analysis of Malware Communities Using Multi-Modal Features*. IEEE Access. 
		8(1):77435-77448. May 2020. https://ieeexplore.ieee.org/document/9076000.
		
2. **Multi-view Modularity Clustering (MVMC)**: An intermediate integration algorithm which works with view graphs of the 
data
	- requires graphs for each view of the data
	- outputs the cluster labels for the objects
	- suitable for large data (up to ~1,000,000 objects) with partially incomplete views


Each algorithm implementation is implemented as scikit-learn clustering module and can be used as such.