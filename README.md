# LinkPredictionGNN
Link Prediction task with multi-hop and embedding neighbours using GraphSAGE.

**Author:** Fabio Tavares

**Email:** fabio.tavares.fma@gmail.com

**LinkedIn:** [Fabiohsst](https://www.linkedin.com/in/fabiohsst/)

If you find this repository helpful, please consider giving it a ⭐

Thanks❗

# Graph Neural Networks for Music Recommendation: Link Prediction

This repository contains the code and resources for a Master's thesis project focused on applying Graph Neural Networks (GNNs) to the task of link prediction in music recommendation systems. The project explores modeling user-music interactions as a heterogeneous graph and using GNNs to predict potential future connections.

## Project Goal

The primary goal of this project was to investigate the effectiveness of Graph Neural Networks for predicting user-item interactions (specifically, user-track and user-artist links) within a music recommendation context. This addresses the challenge of helping users discover new music in a vast content landscape.

## Key Concepts & Technologies

* **Graph Neural Networks (GNNs):** Leveraging the power of neural networks adapted for graph-structured data.
* **Link Prediction:** Framing the recommendation problem as predicting missing or future edges (interactions) in a graph.
* **Heterogeneous Graph:** Modeling the data as a graph with different types of nodes (Users, Tracks, Artists) and edges (e.g., LISTENED\_TO).
* **GraphSAGE:** Implementing a specific GNN architecture known for its inductive capabilities and neighbor aggregation.
* **Music Recommendation Systems (MRS):** Applying advanced ML techniques to improve how music is recommended to users.
* **Dataset:** Utilized the publicly available [LastFM-1k dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) curated by Oscar Celma.
* **Libraries:** Implemented using Python with key libraries such as [PyTorch](https://pytorch.org/), [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/), [Pandas](https://pandas.pydata.org/), and [NumPy](https://numpy.org/).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd [Your Repo Directory Name]
    ```

2.  **Set up a Python environment** (conda, venv, etc. - recommended):
    ```bash
    # Example using conda
    conda create -n gnn-mrec python=3.9
    conda activate gnn-mrec
    ```

3.  **Install dependencies:**
    Install the required Python packages. Make sure to follow the installation instructions for PyTorch and PyTorch Geometric based on your system and CUDA availability if using a GPU.
    ```bash
    # pip install torch torch-geometric pandas numpy scikit-learn [...]
    # Refer to PyTorch Geometric documentation for specific torch/cuda compatible installs
    ```

4.  **Dataset:**
    Download the [LastFM-1k dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html). Place the raw or processed data files in the `data/` directory as expected by the processing scripts.

## Key Findings

This project demonstrated the potential of GNNs for music recommendation via link prediction.

* **User-Artist Prediction:** The GraphSAGE model showed **strong performance** in predicting user-artist connections (e.g., AUC: 98.69%, AP: 97.99%), **outperforming several established benchmark models**.
* **User-Track Prediction:** Predicting user-track links proved **more challenging** with this approach (e.g., AUC: 77.03%, AP: 61.68%), highlighting the distinct complexities of predicting interactions at different granularities.

## Acknowledgements

* Supervisor: [David Leonard] - [Technological University of the Shannon: Midlands Midwest]
