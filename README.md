Awesome-Keyphrase-Prediction
======

This repository is to collect keyphrase prediction resource. 

We strongly encourage the researchers that want to promote their fantastic work to the keyphrase prediction community to make pull request to update their paper's information!


- [Tutorial ](#tutorial)
- [Survey](#survey)
- [Keyphrase Extraction](#keyphrase-extraction)
  - [Unsupervised Extraction](#unsupervised-extraction)
  - [Supervised Extraction](#supervised-extraction)
  - [Multi-Document Extraction](#multi-document-extraction)
- [Keyphrase Generation](#keyphrase-generation)
  - [Model Structure](#model-structure)
    - [Generation paradigm](#generation-paradigm)
      - [One2One](#one2one)
      - [One2Seq](#one2seq)
      - [One2Sst](#one2sst)
    - [Incorporating additional information or constraints](#incorporating-additional-information-or-constraints)
    - [Enhancing Seq2Seq model by GNN \& dual CopyNet](#enhancing-seq2seq-model-by-gnn--dual-copynet)
  - [Training and Decoding](#training-and-decoding)
  - [Absent Keyphrase Generation](#absent-keyphrase-generation)
  - [Multimodal Keyphrase Generation](#multimodal-keyphrase-generation)
  - [Multilingual Keyphrase Generation](#multilingual-keyphrase-generation)
  - [Pre-Training for Generation](#pre-training-for-generation)
  - [Unsupervised Generation](#unsupervised-generation)
  - [Domain Sdaption \& Low Resource](#domain-sdaption--low-resource)
  - [Application of Keyphrase Generation](#application-of-keyphrase-generation)
  - [Empirical Study for Keyphrase Generation](#empirical-study-for-keyphrase-generation)
  - [Other Struggle for Keyphrase Generation](#other-struggle-for-keyphrase-generation)
- [Jointly Extraction and Generation](#jointly-extraction-and-generation)
- [New Dataset](#new-dataset)
- [New Evaluation](#new-evaluation)
- [Chatgpt for Keyphrase Prediction](#chatgpt-for-keyphrase-prediction)


# Tutorial 

-  [A Tutorial on Keyphrasification (ECIR22)](https://keyphrasification.github.io/)


# Survey

- [From statistical methods to deep learning, automatic keyphrase prediction: A survey](https://www.sciencedirect.com/science/article/pii/S030645732300119X)

# Keyphrase Extraction

## Unsupervised Extraction

- [PositionRank: An Unsupervised Approach to KE (ACL17)](https://aclanthology.org/P17-1102)
- [Automatic Ranked KE from Scientific Articles using Phrase Embeddings(NAACL18)](https://aclanthology.org/N18-2100)
- [Simple Unsupervised Keyphrase Extraction using SentenceEmbeddings(CoNLL18)](http://aclanthology.lst.uni-saarland.de/K18-1022.pdf)
- [Unsupervised Keyphrase Extraction with Multipartite Graphs (NAACL18)](https://aclanthology.org/N18-2105)
- [Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context (EMNLP21)](https://aclanthology.org/2021.emnlp-main.14.pdf)
- [AttentionRank: Unsupervised Keyphrase Extraction using Self and Cross Attentions (EMNLP21)](https://aclanthology.org/2021.emnlp-main.146.pdf)
- [Exploiting Position and Contextual Word Embeddings for Keyphrase Extraction from Scientific Papers (EACL21)](https://aclanthology.org/2021.eacl-main.136/)
- [Keyphrase Extraction Using Neighborhood Knowledge Based on Word Embeddings (Arxiv21)](https://arxiv.org/abs/2111.07198)
- [MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction (ACL22)](https://aclanthology.org/2022.findings-acl.34.pdf)
- [PromptRank: Unsupervised Keyphrase Extraction Using Prompt (ACL23)](https://arxiv.org/pdf/2305.04490.pdf)
- [EntropyRank: Unsupervised Keyphrase Extraction via Side-Information Optimization for Language Model-based Text Compression (Arxiv23)](https://arxiv.org/pdf/2308.13399.pdf)



## Supervised Extraction

- [Open Domain Web Keyphrase Extraction Beyond Language Modeling (EMNLP19)](https://aclanthology.org/D19-1521)
- [A Joint Learning Approach based on Self-Distillation for Keyphrase Extraction from Scientific Documents (COLING20)](https://aclanthology.org/2020.coling-main.56/)
- [Keyphrase Extraction with Span-based Feature Representations (Arxiv20)](https://arxiv.org/abs/2002.05407)
- [Joint Keyphrase Chunking and SalienceRanking with BERT (Arxiv20)](https://openreview.net/forum?id=duSg8EGOlX3)
- [Importance Estimation from Multiple Perspectives for Keyphrase Extraction (EMNLP21)](https://aclanthology.org/2021.emnlp-main.215.pdf)
- [Hyperbolic Relevance Matching for Neural Keyphrase Extraction (NAACL22)](https://aclanthology.org/2022.naacl-main.419)
- [Enhancing Phrase Representation by Information Bottleneck Guided Text Diffusion Process for Keyphrase Extraction (Arxiv23)](https://arxiv.org/pdf/2308.08739.pdf)
	
## Multi-Document Extraction

- [Multi-Document Keyphrase Extraction: A Literature Review and the First Dataset (Arxiv21)](https://arxiv.org/abs/2110.01073)

   
# Keyphrase Generation


## Model Structure

### Generation paradigm

#### One2One

- [Deep Keyphrase Generation (ACL17)](https://aclanthology.org/P17-1054)

#### One2Seq

- [One Size Does Not Fit All: Generating and Evaluating Variable Number of Keyphrases (ACL20)](https://aclanthology.org/2020.acl-main.710.pdf)

#### One2Sst

- [One2Set: Generating Diverse Keyphrases as a Set (ACL21)](https://aclanthology.org/2021.acl-long.354)
- [WR-ONE2SET: Towards Well-Calibrated Keyphrase Generation (EMNLP22)](https://arxiv.org/pdf/2211.06862.pdf)

### Incorporating additional information or constraints

- [Keyphrase Generation with Correlation Constraints (EMNLP18)](https://aclanthology.org/D18-1439)
- [Incorporating Linguistic Constraints intoKeyphrase Generation (ACL19)](https://aclanthology.org/P19-1515)
- [Title-Guided Encoding for Keyphrase Generation (AAAI19)](https://ojs.aaai.org/index.php/AAAI/article/view/4587/4465)
- [Topic-Aware Neural Keyphrase Generation for Social Media Language (ACL19)](https://aclanthology.org/P19-1240)
- [SenSeNet: Neural Keyphrase Generation with Document Structure (Arxiv20)](https://arxiv.org/abs/2012.06754)
- [Structure-Augmented Keyphrase Generation (EMNLP21)](https://aclanthology.org/2021.emnlp-main.209.pdf)
- [Keyphrase Generation Beyond the Boundaries of Title and Abstract (Arxiv21)](https://arxiv.org/abs/2112.06776)

### Enhancing Seq2Seq model by GNN & dual CopyNet

- [DivGraphPointer: A Graph Pointer Network for Extracting Diverse Keyphrases (SIGIR19)](https://arxiv.org/abs/1905.07689)
- [Heterogeneous Graph Neural Networks for Keyphrase Generation (EMNLP21)](https://aclanthology.org/2021.emnlp-main.213/)

- [Automatic Keyphrase Generation by Incorporating Dual Copy Mechanisms in Sequence-to-Sequence Learning (COLING22)](https://aclanthology.org/2022.coling-1.204)

## Training and Decoding

- [Semi-Supervised Learning for Neural Keyphrase Generation (EMNLP18)](https://aclanthology.org/D18-1447)
- [Neural Keyphrase Generation via Reinforcement Learningwith Adaptive Rewards (ACL19)](https://aclanthology.org/P19-1208.pdf)
- [Exclusive Hierarchical Decoding for Deep Keyphrase Generation (ACL20)](https://aclanthology.org/2020.acl-main.103)
- [Adaptive Beam Search Decoding for Discrete Keyphrase Generation (AAAI20)](https://ojs.aaai.org/index.php/AAAI/article/view/17546)
- [Diverse Keyphrase Generation with Neural Unlikelihood Training (COLING20)](https://aclanthology.org/2020.coling-main.462/)
- [Keyphrase Generation with Fine-Grained Evaluation-Guided Reinforcement Learning (EMNLP21)](https://aclanthology.org/2021.findings-emnlp.45.pdf)
- [Keyphrase Generation via Soft and Hard Semantic Corrections (EMNLP22)](https://aclanthology.org/2022.emnlp-main.529.pdf)

## Absent Keyphrase Generation
- [Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning (AAAI2022)](https://ojs.aaai.org/index.php/AAAI/article/view/21402)
- [KPDrop: An Approach to Improving Absent Keyphrase Generation (Arxiv21)](https://arxiv.org/abs/2112.01476)


## Multimodal Keyphrase Generation

- [Incorporating Multimodal Information in Open-Domain Web Keyphrase Extraction (EMNLP20)](https://aclanthology.org/2020.emnlp-main.140/)
- [Cross-Media Keyphrase Prediction: A Unified Framework with Multi-Modality Multi-Head Attention and Image Wordings (EMNLP20)](https://aclanthology.org/2020.emnlp-main.268/)

## Multilingual Keyphrase Generation

- [Retrieval-Augmented Multilingual Keyphrase Generation with Retriever-Generator Iterative Training (NAACL22)](https://arxiv.org/abs/2205.10471)

## Pre-Training for Generation

- [Learning Rich Representation of Keyphrases from Text (NAACL22)](https://aclanthology.org/2022.findings-naacl.67/)
- [Applying a Generic Sequence-to-Sequence Model for Simple and Effective Keyphrase Generation (Arxiv22)](https://arxiv.org/abs/2201.05302)

## Unsupervised Generation

- [Unsupervised Deep Keyphrase Generation (AAAI22)](https://ojs.aaai.org/index.php/AAAI/article/view/21381)
- [Unsupervised Open-domain Keyphrase Generation (ACL23)](https://arxiv.org/pdf/2306.10755.pdf)

## Domain Sdaption & Low Resource

- [General-to-Specific Transfer Labeling for Domain Adaptable Keyphrase Generation (Arxiv22)](https://arxiv.org/abs/2208.09606)
- [Representation Learning for Resource-Constrained Keyphrase Generation (EMNLP22)](https://arxiv.org/abs/2203.08118)
- [Data Augmentation for Low-Resource Keyphrase Generation (ACL23)](https://arxiv.org/pdf/2305.17968.pdf)

## Application of Keyphrase Generation
     
- [Keyphrase Generation for Scientific Document Retrieval (ACL20)](https://aclanthology.org/2020.acl-main.105/)
- [Redefining Absent Keyphrases and their Effect on Retrieval Effectiveness (NAACL21)](https://aclanthology.org/2021.naacl-main.330.pdf)

## Empirical Study for Keyphrase Generation

- [An Empirical Study on Neural Keyphrase Generation (NAACL21)](https://aclanthology.org/2021.naacl-main.396.pdf)
- [Pre-trained Language Models for Keyphrase Generation: A Thorough Empirical Study (Arxiv22)](https://arxiv.org/pdf/2212.10233.pdf)


## Other Struggle for Keyphrase Generation

- [Keyphrase Generation: A Text Summarization Struggle (NAACL19)](https://aclanthology.org/N19-1070.pdf)
- [A Preliminary Exploration of GANs for Keyphrase Generation (EMNLP20)](https://aclanthology.org/2020.emnlp-main.645/)

# Jointly Extraction and Generation

- [An Integrated Approach for KG via Exploring the Power of Retrieval and Extraction (NAACL19)](https://aclanthology.org/N19-1292.pdf)
- [Addressing Extraction and Generation Separately: Keyphrase Prediction With Pre-Trained Language Models (TASLP21)](https://ieeexplore.ieee.org/abstract/document/9576585/)
- [SGG: Learning to Select, Guide, and Generate for Keyphrase Generation (NAACL21)](https://aclanthology.org/2021.naacl-main.455.pdf)
- [Select, Extract and Generate: Neural Keyphrase Generation with Layer-wise Coverage Attention (ACL21)](https://aclanthology.org/2021.acl-long.111/)
- [UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction (ACL21)](https://arxiv.org/abs/2106.04847)

# New Dataset

- [Keyphrase Prediction from Video Transcripts: New Dataset and Directions (COLING22)](https://aclanthology.org/2022.coling-1.624.pdf)
	
- [LipKey: A Large-Scale News Dataset for Absent Keyphrases Generation and Abstractive Summarization (COLING22)](https://aclanthology.org/2022.coling-1.303/)
	
- [A new dataset for multilingual keyphrase generation (NIPS22)](https://openreview.net/pdf?id=47qVX2pa-2)

- [A Large-Scale Dataset for Biomedical Keyphrase Generation (LOUHI 2022)](https://arxiv.org/pdf/2211.12124.pdf)

# New Evaluation

- [KPEval: Towards Fine-grained Semantic-based Evaluation of
Keyphrase Extraction and Generation Systems (Arxiv23)](https://arxiv.org/pdf/2303.15422.pdf)
	
# Chatgpt for Keyphrase Prediction

- [Is ChatGPT A Good Keyphrase Generator? A Preliminary Study (Arxiv23)](https://arxiv.org/pdf/2303.13001.pdf)
- [ChatGPT vs State-of-the-Art Models: A Benchmarking Study in Keyphrase Generation Task (Arxiv23)](https://arxiv.org/pdf/2304.14177.pdf)
