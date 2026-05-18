compile = True
max_iters = 800
lr_decay_iters = 800
eval_interval = 100
log_interval = 10
eval_iters = 20

train_data_file = "dataset/train"
valid_data_file = "dataset/validation"
out_dir = "out_finetuned/L4"

# Архитектура (n_layer=4)
n_layer = 4
n_head = 5
n_embd = 160
block_size = 256
bias = False
dropout = 0.0

# Режим загрузки
init_from = "resume"

# Оптимизатор
learning_rate = 3e-5
min_lr = 3e-6
warmup_iters = 100
batch_size = 256
gradient_accumulation_steps = 4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# Системное
device = "cuda"
dtype = "bfloat16"