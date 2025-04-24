class Config:
    TARGET_COLUMNS = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
        'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    PRETRAIN_CHOICE = 1
    MIN_MAX_SCALE = True
    FEATURE_COMPRESS = None
    EMBED_SIZE = 768
    FINGER_SIZE = 167
    NUM_HEADS = 8
    FF_DIM = 256
    DROPOUT = 0.3
    HIDDEN_SIZE = 512
    BATCH_SIZE = 32
    WEIGHT_DECAY = 3e-3  # Regularization to prevent overfitting, useful regardless of phases
    
    # training parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    INCLUDE_BASE_MODEL_PARAMS = True  # When True, trains the base model. When False, it does not update the weights of the base model.
    
    # Other parameters
    PATIENCE = 5
    MAX_POSITION_EMBEDDINGS = 512
    NUM_SMILES_VARIANTS = 10
    MAX_SMILES_VARIANTS_ATTEMPTS = 30
    TOKENIZER = None
    LABEL_SMOOTHING = 0.1
    SIGMA_L2_REG = 1e-4
    ATTENTION_DIM = 128
    
    # PCGrad configuration
    USE_PCGRAD = True  # Whether to use PCGrad
    PCGRAD_REDUCTION = 'mean'  # How to combine projected gradients ('mean' or 'sum')