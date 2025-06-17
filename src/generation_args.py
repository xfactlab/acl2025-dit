def default_args(parser):
    parser.add_argument("--model_name", default="microsoft/phi-2", type=str)
    parser.add_argument("--checkpoint_dir", default='/projects/DMG/saved/llama2', type=str)
    # parser.add_argument("--device_map", default=0, type=int)
    parser.add_argument("--with_pause", action='store_true')
    parser.add_argument("--pause_threshold", default=0.1, type=float)
    parser.add_argument("--original_paper", action='store_true')
    parser. add_argument("--dataset", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="/userhomes/eunki/dynamic_erasing/data/aqua_rat/aqua_rat_test_ver1.json", type=str)
    parser.add_argument("--save_dir", default="/userhomes/eunki/dynamic_erasing/new_predict/", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    return args