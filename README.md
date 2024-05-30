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
		- Cruickshank,  I.J. *Multi-view  Clustering  of  Social-based  Data*.  Ph.D. thesis, CMU-ISR-20-109. Carnegie Mellon University (7,2020)
		- Cruickshank I., Carley K. M. *Analysis of Malware Communities Using Multi-Modal Features*. IEEE Access. 
		8(1):77435-77448. May 2020. https://ieeexplore.ieee.org/document/9076000.
		
2. **Multi-view Modularity Clustering (MVMC)**: An intermediate integration algorithm which works with view graphs of the 
data
	- requires graphs for each view of the data
	- outputs the cluster labels for the objects
	- suitable for large data (up to ~1,000,000 objects) with partially incomplete views
	- references for its use:
		- Cruickshank,  I.J. *Multi-view  Clustering  of  Social-based  Data*.  Ph.D. thesis, CMU-ISR-20-109. Carnegie Mellon University (7,2020)
		- Cruickshank, I. J., & Carley, K. M. (2020). *Characterizing communities of hashtag usage on twitter during the 2020 COVID-19 pandemic by multi-view clustering* Applied Network Science, 5(1), 66.
	- citation:
		```
  		@phdthesis{cruickshank2020multi,
  		title={Multi-view Clustering of Social-based Data.},
  		author={Cruickshank, Iain},
		year={2020},
		school={Carnegie Mellon University, USA}
		}
		```

Each algorithm implementation is implemented as scikit-learn clustering module and can be used as such.