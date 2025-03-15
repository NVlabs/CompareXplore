
# CompareXplore

CompareXplore is a novel approach for design space exploration (DSE) in High-Level Synthesis (HLS) that leverages comparative learning to effectively navigate the pragma design space and identify high-quality hardware designs.

## Overview

CompareXplore introduces several key innovations:

1. A hybrid loss function combining pairwise preference learning with pointwise performance prediction
2. A NODE DIFFERENCE ATTENTION module that focuses on the most informative differences between designs
3. A two-stage DSE approach balancing exploration and exploitation


## Usage

To run CompareXplore:

```bash
cd src
# Modify the config.py to set up configurations.
python main.py
```

Configuration options can be set in `src/config.py`.

## Project Structure

```
.
├── dse_database/         # Dataset and graph representations
├── src/
│   ├── config.py         # Configuration settings
│   ├── main.py           # Main entry point
│   ├── model.py          # Model architecture
│   └── dse.py            # Design Space Exploration logic
├── requirements.txt
└── README.md
```

## Results

Experimental results are saved under  `src/logs/<log_folder>`s.

## Citation

If you use CompareXplore in your research, please cite our paper:

```
@inproceedings{bai2024compareexplore,
  title={Learning to Compare Hardware Designs for High-Level Synthesis},
  author={Bai, Yunsheng and Sohrabizadeh, Atefeh and Ding, Zijian and Liang, Rongjian and Li, Weikai and Wang, Ding and Ren, Haoxing and Sun, Yizhou and Cong, Jason},
  booktitle={MLCAD},
  year={2024}
}
```

