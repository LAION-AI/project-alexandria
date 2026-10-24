"""Standard aliases, usable only when the full term occurs in this source.

An explicit conflicting/ambiguous paper definition always takes precedence.
This table supplies terminology, not an assertion that every acronym occurrence
has that sense. The conservative test arm excludes all such acronym glosses.
"""
ALIASES={
 'DNA':['deoxyribonucleic acid'], 'RNA':['ribonucleic acid'],
 'mRNA':['messenger ribonucleic acid'], 'cDNA':['complementary deoxyribonucleic acid'],
 'PCR':['polymerase chain reaction'], 'qPCR':['quantitative polymerase chain reaction'],
 'ATP':['adenosine triphosphate'],'ADP':['adenosine diphosphate'],
 'ELISA':['enzyme-linked immunosorbent assay'],
 'MRI':['magnetic resonance imaging'],'fMRI':['functional magnetic resonance imaging'],
 'NMR':['nuclear magnetic resonance'],'EEG':['electroencephalography'],
 'XRD':['X-ray diffraction'],'XPS':['X-ray photoelectron spectroscopy'],
 'DFT':['density functional theory','discrete Fourier transform'],
 'FFT':['fast Fourier transform'],'DCT':['discrete cosine transform'],
 'PCA':['principal component analysis'],'ICA':['independent component analysis'],
 'SVM':['support vector machine'],'CNN':['convolutional neural network'],
 'RNN':['recurrent neural network'],'GNN':['graph neural network'],
 'LSTM':['long short-term memory'],'MLP':['multilayer perceptron'],
 'ML':['machine learning'],'AI':['artificial intelligence'],
 'NLP':['natural language processing'],'SGD':['stochastic gradient descent'],
 'MSE':['mean squared error','mean square error'],'RMSE':['root mean squared error'],
 'MAE':['mean absolute error'],'FDR':['false discovery rate'],
 'FPR':['false positive rate'],'TPR':['true positive rate'],
 'FNR':['false negative rate'],'ROC':['receiver operating characteristic'],
 'CPU':['central processing unit'],'GPU':['graphics processing unit'],
 'RAM':['random access memory'],'ODE':['ordinary differential equation'],
 'PDE':['partial differential equation'],'SDE':['stochastic differential equation'],
 'QCD':['quantum chromodynamics'],'QED':['quantum electrodynamics'],
 'QM':['quantum mechanics'],'GR':['general relativity'],
 'UV':['ultraviolet'],'IR':['infrared'],
 'RMS':['root mean square'],'WES':['whole exome sequencing'],'WGS':['whole genome sequencing'],
 'NGS':['next generation sequencing'],'ChIP':['chromatin immunoprecipitation'],
 'ChIP-Seq':['chromatin immunoprecipitation sequencing'],
 'RT-PCR':['reverse transcription polymerase chain reaction'],
 'qRT-PCR':['quantitative reverse transcription polymerase chain reaction'],
 'FBS':['fetal bovine serum'],'PBS':['phosphate buffered saline'],
 'HPLC':['high performance liquid chromatography'],'TLC':['thin layer chromatography'],
 'SEM':['scanning electron microscopy'],'TEM':['transmission electron microscopy'],
 'HRTEM':['high resolution transmission electron microscopy'],
 'XANES':['X-ray absorption near edge structure'],'EXAFS':['extended X-ray absorption fine structure'],
 'SPH':['smoothed particle hydrodynamics'],'FDTD':['finite difference time domain'],
 'BBN':['broad band noise'],'QPO':['quasi periodic oscillation'],
 'GWAS':['genome wide association study'],'SNP':['single nucleotide polymorphism'],
 'ROI':['region of interest'],'ROIs':['regions of interest'],
}
for key in ['SVM','CNN','RNN','GNN','MLP','CPU','GPU','ODE','PDE','SDE']:
    ALIASES[key+'s']=[value+'s' for value in ALIASES[key]]
