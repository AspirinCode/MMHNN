# MMHNN

**Molecular Merged Hypergraph Neural Network for Explainable Solvation Free Energy Prediction**

Solvation free energies play a fundamental role in various fields of chemistry and biology. Accurately determining the solvation Gibbs free energy (∆Gsolv) of a molecule in a given solvent requires a deep understanding of the intrinsic relationships between solute and solvent molecules. While deep learning methods have been developed for ∆Gsolv prediction, few explicitly model intermolecular interactions between solute and solvent molecules. MMGNN more closely aligns with real-world chemical processes by explicitly capturing atomic-level interactions, such as hydrogen bonding. It achieves this by initially establishing indiscriminate connections between intermolecular atoms, which are subsequently refined using an attention-based aggregation mechanism tailored to specific solute-solvent pairs. However, its sharply increasing computational complexity limits its scalability and broader applicability. Here, we introduce an improved framework, Molecular Merged Hypergraph Neural Network (MMHNN), which leverages a predefined subgraph set and replaces subgraphs with supernodes to construct a hypergraph representation. This design effectively mitigates model complexity while preserving key molecular interactions. Furthermore, to handle non-interactive or repulsive atomic interactions, MMHNN incorporates an interpretation mechanism for nodes and edges within the merged graph, leveraging Graph Information Bottleneck (GIB) theory to enhance model explainability. Extensive experimental validation demonstrates the efficiency of MMHNN and its improved interpretability in capturing solute-solvent interactions.





## Cite


* Wenjie Du, Shuai Zhang, Zhaohui Cai, Zhiyuan Liu, Junfeng Fang, Jianmin Wang, Yang Wang. Molecular Merged Hypergraph Neural Network for Explainable Solvation Free Energy Prediction. Research. 0:DOI:10.34133/research.0740





