def default_args(parser):
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--save_dir", default='/projects/DMG/saved/phi1_5', type=str)
    parser.add_argument("--predict_dir", default='./predict', type=str)
    parser.add_argument("--data_name", default='gsm8k', type=str)
    parser.add_argument("--train_data_dir", default=None, type=str)
    parser.add_argument("--test_data_dir", default=None, type=str)
    parser.add_argument("--model_name", default="microsoft/phi-2", type=str)

    # Training Arguments
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--torch_compile", default=False, type=bool)
    parser.add_argument("--flash_attention_2", action='store_true')
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--optim", default="paged_adamw_32bit", type=str)
    parser.add_argument("--overwrite_output_dir", default=True, type=bool)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--per_device_train_batch_size", default=2, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=2, type=int)
    parser.add_argument("--warmup_steps", default=5000, type=int)    
    parser.add_argument("--evaluation_strategy", default='epoch', type=str)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--save_strategy", default='epoch', type=str)
    parser.add_argument("--prompt_max_length", default=256, type=int)
    parser.add_argument("--response_max_length", default=1024, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    
    parser.add_argument("--mask_type", default="pause", type=str)
    parser.add_argument("--pause_type", default=None, type=str)
    parser.add_argument("--pause_num", default=1, type=int)
    parser.add_argument("--dynamic_num", default=1, type=int)
    parser.add_argument("--delete_second_math", action='store_true')
    parser.add_argument("--is_dynamic", action='store_true')
    
    parser.add_argument("--not_pause_learning", action='store_true')
    
    parser.add_argument("--run_name", default="", type=str)
    
    

    # LoRA
    parser.add_argument("--enable_lora", action='store_true')
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_rank", default=64, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=int)


    # Wandb Configurations
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project_name", default=None, type=str)

    parser.add_argument("--train_subset", action='store_true')
    
    args = parser.parse_args()

    return args