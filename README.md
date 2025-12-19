# Gluco-LLM: Glucose Time Series Forecasting with Large Language Models

[![Paper](https://img.shields.io/badge/Paper-CSB%20Reports-blue)](https://doi.org/10.1016/j.csbr.2025.100068)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Official implementation of the research article:

> **LLM-powered personalized glucose prediction in type 1 diabetes**  
> Qingrui Li, Kapileshwor Ray Amat, Juan Li  
> *Computational and Structural Biotechnology Reports*, 2025. DOI: 10.1016/j.csbr.2025.100068

---

## Overview

**Gluco-LLM** is a personalized glucose forecasting framework powered by Large Language Models (LLMs). It leverages LLMs (GPT2, LLAMA, Deepseek) to predict future glucose levels based on historical measurements and additional features like insulin injections and meal information. This project extends the [Time-LLM](https://github.com/KimMeen/Time-LLM) framework to specifically handle glucose time series forecasting.

---

## Features

- **Multi-Model Support**: Compatible with GPT2, LLAMA, and Deepseek models
- **Glucose-Specific Features**: Handles historical measurements and additional features like insulin injections and meal information
- **Time Series Forecasting**: Predicts glucose levels for diabetes management
- **Flexible Architecture**: Supports both training and inference modes
- **Comprehensive Evaluation**: Includes MAE, RMSE, and other metrics (e.g. MARD)

---

## Project Demo

The video file is located in the `Demo/` directory of this repository.

**Direct link:** [Watch Demo Video](./Demo/Demo%20Video.mov)

---

## Getting Started

### 1) Installation

```bash
git clone https://github.com/Intelligent-System-Lab/Gluco-LLM.git
cd Gluco-LLM
pip install -r requirements.txt
```

---

## Data Access (OhioT1DM)

This project is benchmarked on the **OhioT1DM** dataset.

- **Important:** The OhioT1DM dataset is **not included** in this repository due to access/licensing restrictions.
- **Official request link:** https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
- Access may require signing a data use agreement with Ohio University.

---

## Data Format

The code expects glucose data in CSV format with (at minimum) the following columns:

- `ts`: timestamp
- `glucose`: target glucose values
- additional feature columns as available (e.g., insulin injection data, meal information, etc.)

---

## Usage

The repo supports both **training (personalization)** and **testing/inference** modes.

### Training (Personalization) Example

```bash
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 588_train.csv \
  --test_data_path 588_test.csv \
  --model_id 588 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len $s_len \
  --label_len $l_len \
  --pred_len $p_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
```

### Testing / Inference Example

```bash
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 0 \
  --separate_test yes \
  --root_path ./dataset/glucose/ \
  --data_path 588_train.csv \
  --test_data_path 588_test.csv \
  --model_id 588 \
  --model $model_name \
  --data Glucose \
  --features S \
  --seq_len $s_len \
  --label_len $l_len \
  --pred_len $p_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
```

---

## Results

The model outputs predictions in the `./results/` directory with detailed metrics including:
- Average prediction error
- Last prediction error
- MAE and RMSE metrics

---

## Configuration

Key parameters:
- `--seq_len`: Input sequence length
- `--pred_len`: Prediction horizon
- `--llm_model`: Choice of LLM (GPT2, LLAMA, BERT)
- `--llm_layers`: Number of LLM layers to use

---

## Citation

If you find this code or research useful, please cite:

```bibtex
@article{li2025llm,
  title={LLM-powered personalized glucose prediction in type 1 diabetes},
  author={Li, Qingrui and Amat, Kapileshwor Ray and Li, Juan},
  journal={Computational and Structural Biotechnology Reports},
  volume={2},
  pages={100068},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.csbr.2025.100068}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgement

- This repository adapts code from [Time-LLM](https://github.com/KimMeen/Time-LLM), which is licensed under the Apache License, Version 2.0. We thank the original authors for providing the codebase.

---

## Contributing & Contact

Contributions are welcome! Please feel free to submit a Pull Request.
For questions and issues, please open an issue on GitHub. 
