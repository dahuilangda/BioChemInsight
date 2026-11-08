<div align="center">

<h1>MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning</h1>

AAAI 2026 Accepted Paper

Wenrui Zhang<sup>1</sup> · Xinggang Wang<sup>1</sup> · Bin Feng<sup>1</sup> · Wenyu Liu<sup>1</sup>

<sup>1</sup>School of Electronic Information and Communications, Huazhong University of Science and Technology

<a href="https://arxiv.org/pdf/2511.17300"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a> <a href="https://github.com/hustvl/MolSight"><img src="https://img.shields.io/badge/Code-GitHub-black.svg" alt="Code"></a> <a href="https://github.com/hustvl/MolSight/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>

</div>

## 📖 Introduction

**MolSight** is a comprehensive learning framework for Optical Chemical Structure Recognition (OCSR), designed to bridge the gap between computer vision and chemical informatics (AI4S).

Accurately translating molecular images into machine-readable formats (like SMILES) is critical for drug discovery and digital chemistry. MolSight addresses the limitations of previous methods—particularly in handling complex **stereoisomers**—through a novel three-stage training paradigm:

1.  **SMILES Pretraining:** Aligns visual representations with chemical strings.
2.  **Multi-Granularity Fine-Tuning:** Captures both global structure and local functional group details.
3.  **RL Post-Training:** Utilizes Reinforcement Learning to optimize for chemical semantic correctness rather than simple token matching.

### ✨ Key Features

  * **First RL-based OCSR:** MolSight is the first OCSR system to integrate **Reinforcement Learning**. We utilize Group Relative Policy Optimization (GRPO) to directly optimize chemical validity[c.
  * **Stereo-200k Dataset:** We introduce a new annotated dataset consisting of **200,000 challenging stereoisomeric molecules** specifically curated to address confusion in 3D chiral structures.

<img width="966" height="564" alt="image" src="https://github.com/user-attachments/assets/d6bbe0b9-d890-4f5b-99f7-f11601a4be65" />

  * **SOTA Performance:** Extensive experiments demonstrate that MolSight achieves state-of-the-art results in accuracy, similarity, and robustness, outperforming classical and learning-based baselines.

## 🔥 News

  * **[2025-11-26]** 🎉 MolSight has been accepted to **AAAI 2026**\!
  * **[2025-11-26]** 🚀 Code released.

## Updates

- [x] Release code
- [x] Release Stereo-200k dataset
- [x] Release model weights
- [x] Release inference demo

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/hustvl/MolSight
cd MolSight

# Install dependencies
pip install -r requirements.txt
```

### Inference Demo
```bash
python inference.py
```

## Data
### Training Datasets
1. Pretrain dataset: [MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M)
2. SFT datasets: [PubChem-1M](https://huggingface.co/yujieq/MolScribe/blob/main/pubchem.zip), [USPTO-680k](https://huggingface.co/yujieq/MolScribe/blob/main/uspto_mol.zip)
3. RL dataset: [Stereo-200k](https://huggingface.co/datasets/Robert-zwr/Stereo-200k)
### Evaluation Datasets
+ USPTO, UoB, CLEF, JPO: [images](https://github.com/Kohulan/OCSR_Review/tree/master/assets/images), [labels](https://github.com/Kohulan/OCSR_Review/tree/master/assets/reference), we also provided [labels in SMILES format](https://github.com/hustvl/MolSight/tree/main/data/real).
+ [Stereo-2k](https://huggingface.co/datasets/Robert-zwr/Stereo-200k)

**Notes:**
The Stereo dataset is introduced for the first time in this work, consisting entirely of stereoisomeric molecules.

## Weights
<table class="center">
<tr>
  <td style="text-align:center;"><b>Name</b></td>
  <td style="text-align:center;"><b>Predict Field</b></td>
  <td style="text-align:center;"><b>Description</b></td>
  <td style="text-align:center;"><b>Acc. on USPTO</b></td>
</tr>
    
<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/Robert-zwr/MolSight/blob/main/pubchem_uspto_smiles_edges_10.pth">MolSight-base</a></td>
  <td style="text-align:center;"><b>SMILES & edge</b></td>
  <td style="text-align:center;"><b>Trained on PubChem-1M and USPTO-680k for 10 epochs.</b></td>
  <td style="text-align:center;"><b>91.2</b></td>
</tr>
    
<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/Robert-zwr/MolSight/blob/main/pubchem_coords_10_2.pth">MolSight-coord</a></td>
  <td style="text-align:center;"><b>SMILES & edge & coord</b></td>
  <td style="text-align:center;"><b>Continue trained on PubChem-1M for 2 epochs to get a coord head.</b></td>
  <td style="text-align:center;"><b>91.1</b></td>
</tr>

<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/Robert-zwr/MolSight/blob/main/stereo_grpo_10_2.pth">MolSight-stereo</a></td>
  <td style="text-align:center;"><b>SMILES</b></td>
  <td style="text-align:center;"><b>Continue trained on Stereo-200k with LoRA for 2 epochs to get better performance on stereo molecules.</b></td>
  <td style="text-align:center;"><b>90.3</b></td>
</tr>

<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/Robert-zwr/MolSight/blob/main/pubchem_uspto_smiles_edges_30.pth">MolSight-extra</a></td>
  <td style="text-align:center;"><b>SMILES & edge</b></td>
  <td style="text-align:center;"><b>Similar to MolSight-base, but with extra training steps (30 epochs), usually can get better evaluation score.</b></td>
  <td style="text-align:center;"><b>92.0</b></td>
</tr>

<tr>
  <td style="text-align:center;"><a href="https://huggingface.co/Robert-zwr/MolSight/blob/main/epoch_49.pth">MolSight-Markush</a></td>
  <td style="text-align:center;"><b>SMILES</b></td>
  <td style="text-align:center;"><b>Finetuned on <a href="https://huggingface.co/datasets/docling-project/MarkushGrapher-Datasets">MarkushGrapher</a>, can predict SMILES-M to deal with Markush structures.</b></td>
  <td style="text-align:center;"><b>-</b></td>
</tr>
</table>

## Training

Start MolSight training with:

```bash
# SFT
bash train.sh
# train the additional coord predictor
bash train_loc_predictor.sh
# post training with RL
bash post_train.sh
```

## Citation

If you find MolSight or the Stereo-200k dataset useful for your research in AI4Science or Chemistry, please cite our paper:

```bibtex
@article{zhang2025molsight,
  title={MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning},
  author={Zhang, Wenrui and Wang, Xinggang and Feng, Bin and Liu, Wenyu},
  journal={arXiv preprint arXiv:2511.17300},
  year={2025}
}
```

## Acknowledgement

This project has referenced some excellent open-sourced repos ([MolScribe](https://github.com/thomas0809/MolScribe), [trl](https://github.com/huggingface/trl), [Whisper](https://github.com/openai/whisper), [MMPose](https://github.com/open-mmlab/mmpose)). Thanks for their wonderful works and contributions to the community.

