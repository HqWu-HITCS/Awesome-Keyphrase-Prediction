Keyphrase- Prediction-Resource
======

This repository is to collect keyphrase prediction resource. 

- [Keyphrase- Prediction-Resource](#keyphrase--prediction-resource)
  - [Tutorial](#tutorial)
  - [Keyphrase Extraction Paper](#keyphrase-extraction-paper)
    - [Unsupervised ranking](#unsupervised-ranking)
    - [Supervised extraction methods](#supervised-extraction-methods)
    - [Multi- Document Keyphrase Extraction](#multi--document-keyphrase-extraction)
  - [Keyphrase Generation Paper](#keyphrase-generation-paper)
    - [Generation paradigm:](#generation-paradigm)
    - [Model structure:](#model-structure)
    - [Enhancing model by more information/Constraints](#enhancing-model-by-more-informationconstraints)
    - [Training and decoding](#training-and-decoding)
    - [Jointly extractive and a generative for keyphrase](#jointly-extractive-and-a-generative-for-keyphrase)
    - [Multimodal keyphrase generation](#multimodal-keyphrase-generation)
    - [Multilingual keyphrase generation](#multilingual-keyphrase-generation)
    - [Pre-training model for keyphrase generation](#pre-training-model-for-keyphrase-generation)
    - [GNN for keyphrase generation](#gnn-for-keyphrase-generation)
    - [Unsupervised keyphrase generation](#unsupervised-keyphrase-generation)
    - [Domain adaption](#domain-adaption)
    - [Application](#application)
    - [Otherwork for keyphrase generation](#otherwork-for-keyphrase-generation)
  - [New Dataset](#new-dataset)

## Tutorial 

-  [A Tutorial on Keyphrasification ECIR22](https://keyphrasification.github.io/)



## Keyphrase Extraction Paper

### Unsupervised ranking

- [PositionRank: An Unsupervised Approach to KE (ACL17)](https://aclanthology.org/P17-1102)
- [Automatic Ranked KE from Scientific Articles using Phrase Embeddings(NAACL18)](https://aclanthology.org/N18-2100)
- [Simple Unsupervised Keyphrase Extraction using SentenceEmbeddings(CoNLL18)](http://aclanthology.lst.uni-saarland.de/K18-1022.pdf)
- [Unsupervised Keyphrase Extraction with Multipartite Graphs (NAACL18)](https://aclanthology.org/N18-2105)
- [Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context (EMNLP21)](https://aclanthology.org/2021.emnlp-main.14.pdf)
- [AttentionRank: Unsupervised Keyphrase Extraction using Self and Cross Attentions (EMNLP21)](https://aclanthology.org/2021.emnlp-main.146.pdf)
- [MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction (ACL22)](https://aclanthology.org/2022.findings-acl.34.pdf)
- [Keyphrase Extraction Using Neighborhood Knowledge Based on Word Embeddings (Arxiv21)](https://arxiv.org/abs/2111.07198)


### Supervised extraction methods

- [Open Domain Web Keyphrase Extraction Beyond Language Modeling (EMNLP19)](https://aclanthology.org/D19-1521)
- [Keyphrase Extraction with Span-based Feature Representations (Arxiv20)](https://arxiv.org/abs/2002.05407)
- [Joint Keyphrase Chunking and SalienceRanking with BERT (Arxiv20)](https://openreview.net/forum?id=duSg8EGOlX3)
- [Importance Estimation from Multiple Perspectives for Keyphrase Extraction (EMNLP21)](https://aclanthology.org/2021.emnlp-main.215.pdf)
- [Hyperbolic Relevance Matching for Neural Keyphrase Extraction (NAACL22)](https://aclanthology.org/2022.naacl-main.419)

	
### Multi- Document Keyphrase Extraction

- [Multi-Document Keyphrase Extraction: A Literature Review and the First Dataset (Arxiv21)](https://arxiv.org/abs/2110.01073)

   
## Keyphrase Generation Paper

### Generation paradigm:

- [One2One: Deep Keyphrase Generation (ACL17)](https://aclanthology.org/P17-1054)
- [One2Seq: One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases (ACL20)](https://aclanthology.org/2020.acl-main.710.pdf)
- [One2Set: Generating Diverse Keyphrases as a Set (ACL21)](https://aclanthology.org/2021.acl-long.354)

### Model structure:

- [Automatic Keyphrase Generation by Incorporating Dual Copy Mechanisms in Sequence-to-Sequence Learning (COLING22)](https://aclanthology.org/2022.coling-1.204)

### Enhancing model by more information/Constraints

- [Keyphrase Generation with Correlation Constraints (EMNLP18)](https://aclanthology.org/D18-1439)
- [Incorporating Linguistic Constraints intoKeyphrase Generation (ACL19)](https://aclanthology.org/P19-1515)
- [Title-Guided Encoding for Keyphrase Generation (AAAI19)](https://ojs.aaai.org/index.php/AAAI/article/view/4587/4465)
- [Topic-Aware Neural Keyphrase Generation for Social Media Language (ACL19)](https://aclanthology.org/P19-1240)
- [SenSeNet: Neural Keyphrase Generation with Document Structure (Arxiv20)](https://arxiv.org/abs/2012.06754)
- [Structure-Augmented Keyphrase Generation (EMNLP21)](https://aclanthology.org/2021.emnlp-main.209.pdf)
- [Keyphrase Generation Beyond the Boundaries of Title and Abstract (Arxiv21)](https://arxiv.org/abs/2112.06776)

### Training and decoding

- [Semi-Supervised Learning for Neural Keyphrase Generation (EMNLP18)](https://aclanthology.org/D18-1447)
- [Neural Keyphrase Generation via Reinforcement Learningwith Adaptive Rewards (ACL19)](https://aclanthology.org/P19-1208.pdf)
- [Exclusive Hierarchical Decoding for Deep Keyphrase Generation (ACL20)](https://aclanthology.org/2020.acl-main.103)
- [Adaptive Beam Search Decoding for Discrete Keyphrase Generation (AAAI20)](https://ojs.aaai.org/index.php/AAAI/article/view/17546)
- [Keyphrase Generation with Fine-Grained Evaluation-Guided Reinforcement Learning (EMNLP21)](https://aclanthology.org/2021.findings-emnlp.45.pdf)
- [Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning (AAAI2022)](https://ojs.aaai.org/index.php/AAAI/article/view/21402)

### Jointly extractive and a generative for keyphrase

- [An Integrated Approach for KG via Exploring the Power of Retrieval and Extraction (NAACL19)](https://aclanthology.org/N19-1292.pdf)
- [Addressing Extraction and Generation Separately: Keyphrase Prediction With Pre-Trained Language Models (TASLP21)](https://ieeexplore.ieee.org/abstract/document/9576585/)
- [SGG: Learning to Select, Guide, and Generate for Keyphrase Generation (NAACL21)](https://aclanthology.org/2021.naacl-main.455.pdf)
- [Select, Extract and Generate: Neural Keyphrase Generation with Layer-wise Coverage Attention (ACL21)](https://aclanthology.org/2021.acl-long.111/)
- [UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction (ACL21)](https://arxiv.org/abs/2106.04847)

### Multimodal keyphrase generation

- [Incorporating Multimodal Information in Open-Domain Web Keyphrase Extraction (EMNLP20)](https://aclanthology.org/2020.emnlp-main.140/)
- [Cross-Media Keyphrase Prediction: A Unified Framework with Multi-Modality Multi-Head Attention and Image Wordings (EMNLP20)](https://aclanthology.org/2020.emnlp-main.268/)

### Multilingual keyphrase generation

- [Retrieval-Augmented Multilingual Keyphrase Generation with Retriever-Generator Iterative Training (NAACL22)](https://arxiv.org/abs/2205.10471)

### Pre-training model for keyphrase generation

- [Learning Rich Representation of Keyphrases from Text (NAACL22)](https://aclanthology.org/2022.findings-naacl.67/)
- [Applying a Generic Sequence-to-Sequence Model for Simple and Effective Keyphrase Generation (Arxiv22)](https://arxiv.org/abs/2201.05302)

### GNN for keyphrase generation

- [DivGraphPointer: A Graph Pointer Network for Extracting Diverse Keyphrases (SIGIR19)](https://arxiv.org/abs/1905.07689)
- [Heterogeneous Graph Neural Networks for Keyphrase Generation (EMNLP21)](https://aclanthology.org/2021.emnlp-main.213/)

### Unsupervised keyphrase generation

- [Unsupervised Deep Keyphrase Generation (AAAI22)](https://ojs.aaai.org/index.php/AAAI/article/view/21381)

### Domain adaption

- [General-to-Specific Transfer Labeling for Domain Adaptable Keyphrase Generation (Arxiv22)](https://arxiv.org/abs/2208.09606)

### Application
     
- [Keyphrase Generation for Scientific Document Retrieval (ACL20)](https://aclanthology.org/2020.acl-main.105/)
- [Redefining Absent Keyphrases and their Effect on Retrieval Effectiveness (NAACL21)](https://aclanthology.org/2021.naacl-main.330.pdf)

### Otherwork for keyphrase generation

- [Keyphrase Generation: A Text Summarization Struggle (NAACL19)](https://aclanthology.org/N19-1070.pdf)
- [A Preliminary Exploration of GANs for Keyphrase Generation (EMNLP20)](https://aclanthology.org/2020.emnlp-main.645/)
- [An Empirical Study on Neural Keyphrase Generation (NAACL21)](https://aclanthology.org/2021.naacl-main.396.pdf)
- [KPDrop: An Approach to Improving Absent Keyphrase Generation (Arxiv21)](https://arxiv.org/abs/2112.01476)

## New Dataset

- [Keyphrase Prediction from Video Transcripts: New Dataset and Directions (COLING222)](https://aclanthology.org/2022.coling-1.624.pdf)
	
- [LipKey: A Large-Scale News Dataset for Absent Keyphrases Generation and Abstractive Summarization (COLING222)](https://aclanthology.org/2022.coling-1.303/)
	
- [A new dataset for multilingual keyphrase generation (NIPS22)](https://openreview.net/pdf?id=47qVX2pa-2)
