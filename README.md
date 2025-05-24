# Efficient Robotic Policy Learning via Latent Space Backward Planning
[[Website](https://dstate.github.io/LBP/)]  [[Paper](https://www.arxiv.org/pdf/2505.06861)]
🔥 **LBP has been accepted by ICML2025** 🔥

## Introduction

We propose a Latent Space Backward Planning scheme (LBP), which balances efficiency, accuracy and sufficient future guidance. LBP begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating ontask prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance.

<p align="center"> 
	<img src="assets/images/LBP_intro.jpg"width="100%"> 
</p>


## Citation
- If you want to reference this work, please cite it as:
```
@inproceedings{
    liu-niu2025lbp,
    title={Efficient Robotic Policy Learning via Latent Space Backward Planning},
    author={Dongxiu Liu and Haoyi Niu and Zhihao Wang and Jinliang Zheng and Yinan Zheng and zhonghong Ou and Jianming Hu and Jianxiong Li and Xianyuan Zhan},
    booktitle={International Conference on Machine Learning},
    year={2025}
}
```

## Quick Start

### Install

1. Clone this repository and create an environment
```bash
conda create -n lbp python=3.8 -y
conda activate lbp
git clone git@github.com:Dstate/LBP.git
cd LBP
```

2. Set up [DecisionNCE](https://github.com/2toinf/DecisionNCE)
```bash
git clone https://github.com/2toinf/DecisionNCE.git
cd DecisionNCE
pip install -e .
cd ..
```

3. We use the checkpoint of [DecisionNCE(Robo-MUTUAL)](https://github.com/255isWhite/Robo_MUTUAL), please download from [link](https://drive.google.com/file/d/1_bvhXUzWYWhg7bUANhDRB9Zq09wKcjB1/view?usp=drive_link)
```bash
mkdir -p ~/.cache/DecisionNCE
mv <above_downloaded_ckpt> DecisionNCE-T
mv DecisionNCE-T ~/.cache/DecisionNCE
```

4. Set up [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
```

5. Install other package

```bash
pip install -r requirements.txt
```