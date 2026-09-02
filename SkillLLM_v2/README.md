Here we present the accompanying code for the following paper

# Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction

Make sure to install all the requirements first

``` bash
pip install -r requirements.txt
```

If your account is able to download LLaMA 3 models, log into huggingface first

```bash
huggingface-cli login
```

For better loss logging, especially for GLiNER log into WandB

```bash
wandb login
```

## Large Language Model fine tuning

To download the data and preprocess it from the root directory run

``` bash
bash src/data_preparation.sh
```

To run training, validation generation and then get metrics from the root directory run

``` bash
bash src/training.sh
```

## GLiNER

For finetuning GLiNER you can find the relevant code under

```
src/gliner/gliner_raytune.ipynb
```