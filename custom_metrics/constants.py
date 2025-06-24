MODEL = "../models/meta-llama/Meta-Llama-3.1-8B-Instruct"


# Training Params
output_dir = "experiments_5"
num_train_epochs = 20
per_device_train_batch_size = 8
eval_batch_size = 8
gradient_accumulation_steps = 4
gradient_checkpoint = True
max_grad_norm = 0.3
learning_rate = 1e-4
weight_decay = 0.01
optim = "adamw_8bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
warmup_steps = 10
seed = 3047

# SFT Params ?
max_seq_length = 2048
packing = False
device_map = {"": 0}