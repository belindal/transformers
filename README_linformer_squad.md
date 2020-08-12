Convert checkpoint
```console
python src/transformers/convert_linformer_original_pytorch_checkpoint_to_pytorch.py \
    --linformer_checkpoint_path /checkpoint/belindali/linformer/fairseq_checkpoint \
    --pytorch_dump_folder_path /checkpoint/belindali/linformer/hf_checkpoint
```

Run squad
```console
SQUAD_DIR=/checkpoint/belindali/SQUAD

python ./run_squad_linformer.py --model_type linformer --config_name /checkpoint/belindali/linformer/hf_checkpoint \
    --tokenizer_name roberta-base  --model_name_or_path /checkpoint/belindali/linformer/hf_checkpoint \
    --do_train --do_eval --train_file $SQUAD_DIR/train-v1.1.json --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5     --num_train_epochs 2     --max_seq_length 384     --doc_stride 128     --output_dir ../models/linformer_finetuned_squad/
```
