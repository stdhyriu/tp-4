# Configurar o ambiente

Há duas formas de configurar o ambiente, o mais recomendável é o presente no item 2:

1. Para instalar o ambiente, mas não é o mais recomendável

`pip install -r requirements.txt`

2. A biblioteca unsloth ainda é um pouco instável durante o processo de instalação, estes passos garantem que o ambiente fique exatamente igual ao utilizado para este projeto.

`pip install unsloth`

`pip install tensorboardx`

`pip install accelerate==1.7.0`

`pip install triton==3.2.0`

# Pre-Processamento dos Dados

Para gerar os dados de treinamento:

```python
python generate_sql_outputs.py --database_path spider_data/database --table_json spider_data/tables.json --spider_data spider_data/train_spider.json
```

Para gerar os dados de teste:

```python
python generate_sql_outputs.py --database_path spider_data/database --table_json spider_data/tables.json --spider_data spider_data/dev.json
```

# Fine Tuning

Foram realizados dois fine tunings 

1 Finetuning, parâmetros

```python
learning_rate = 1e-4
lora_alpha = 16,
lora_dropout = 0.,
```


2 Finetuning, parâmetros

```python
learning_rate = 1e-5
lora_alpha = 32,
lora_dropout = 0.1,
```


3 Finetuning, parâmetros

```python
learning_rate = 1e-5
lora_alpha = 32,
lora_dropout = 0.1,
epochs=30
```


4 Finetuning, parâmetros

```python
learning_rate = 1e-4
lora_alpha = 16,
r = 8,
lora_dropout = 0.2,
```

5 Finetuning, parâmetros

```python
learning_rate = 1e-4
lora_alpha = 16,
r = 16,
lora_dropout = 0.2,
weight_decay = 0.01,
```

Para realizar o fine-tuning basta digitar:

`python main.py`

# Generalização:

Para gerar os resultados da generalização:

`python mmlu_eval.py`

Para quantificar os dados:

`python apply_metric_mmlu_evaluation.py`


# Validar o fine-tuning dos comando SQL:

`python apply_sql_command_evaluation.py`


## Resultados

Na pasta ``/results`` estão os resultados dos algoritmos:

* ``few_shot_results.csv``: resultados do modelo **Llama 3.1 8B Instruct** sem fine-tuning.

* ``fine_tuning_1_results--checkpoint--1014.csv``: resultados do finetuning realizado no model **Llama 3.1 8B - Instruct**, **experimento 1**.

* ``fine_tuning_2_results--checkpoint--2028.csv``: resultados do finetuning realizado no model **Llama 3.1 8B - Instruct**, **experimento 2**.

* ``fine_tuning_3_results--checkpoint--1521.csv``: resultados do finetuning realizado no model **Llama 3.1 8B - Instruct**, **experimento 3**.

* ``fine_tuning_4_results--checkpoint--1014.csv``: resultados do finetuning realizado no model **Llama 3.1 8B - Instruct**, **experimento 4**.

* ``fine_tuning_5_results--checkpoint--XXXX.csv``: resultados do finetuning realizado no model **Llama 3.1 8B - Instruct**, **experimento 5**.