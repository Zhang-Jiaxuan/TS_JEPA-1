# config/config_downstream.py

config = {
    # Data paths
    "path_train": "./data/FordB/train_data.ts",
    "path_test": "./data/FordB/test_data.ts",
    "path_save": "./logs/output_model/",

    "regression": False,

    "wandb_project_name": "",
    "log_wandb": False,

    # Printing and logging
    "checkpoint_print": 20,

    # Dataset / P0 protocol
    "data": "weather",
    "seed": 42,
    "patch_size": 32,
    "val_fraction": 0.05,
    "test_fraction": 0.30,
    "ratio_patches": 20,
    "ratio_supervision": 1.0,

    # Optimization
    "num_epochs": 101,
    "batch_size": 512,
    "lr": 1e-5,

    # CNN
    "cnn_out_channels": [32, 64, 128],
    "cnn_kernel_size": 3,
    "cnn_dense_dim": 32,

    # Attention only
    "attention_only_dense_dim": 512,
    "attention_only_embed_dim": 512,

    # Transformer
    "embed_dim": 128,
    "nhead": 2,
    "num_layers": 1,
    "kernel_size": 3,
    "embed_bias": True,
    "transformer_dense_dim": 64,
    "pooling": "Mean",

    # Pretrained encoder
    "pretrain_encoder_embed_dim": 256,
    "pretrain_encoder_nhead": 4,
    "pretrain_encoder_num_layers": 2,
    "pretrain_encoder_kernel_size": 3,
    "pretrain_encoder_embed_bias": True,
    "pretrain_transformer_dense_dim": 128,

    # Pretrained decoder
    "pretrain_decoder_embed_dim": 128,
    "pretrain_decoder_nhead": 2,
    "pretrain_decoder_num_layers": 1,

    "checkpoint_to_use": 3000,
    "lr_pretrain": 1e-5,
    "batch_size_pretrain": 256,
    "mask_ratio": 0.9,
    "ema_pretrain": 0.999,
}
