#!/bin/bash

uv run project-2 --epochs 8 --learning_rate 0.0001 --batch_size 64 \
  --num_enc_layers 4 --num_dec_layers 4 --d_model 512 --num_heads 8 \
  --d_ff 1024 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json

uv run project-2 --epochs 16 --learning_rate 0.0001 --batch_size 64 \
  --num_enc_layers 4 --num_dec_layers 4 --d_model 512 --num_heads 8 \
  --d_ff 1024 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json

uv run project-2 --epochs 32 --learning_rate 0.0001 --batch_size 64 \
  --num_enc_layers 4 --num_dec_layers 4 --d_model 512 --num_heads 8 \
  --d_ff 1024 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json

uv run project-2 --epochs 8 --learning_rate 0.0001 --batch_size 64 \
  --num_enc_layers 2 --num_dec_layers 2 --d_model 512 --num_heads 8 \
  --d_ff 1024 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json

uv run project-2 --epochs 16 --learning_rate 0.0001 --batch_size 62 \
  --num_enc_layers 2 --num_dec_layers 2 --d_model 512 --num_heads 8 \
  --d_ff 1022 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json

uv run project-2 --epochs 32 --learning_rate 0.0001 --batch_size 62 \
  --num_enc_layers 2 --num_dec_layers 2 --d_model 512 --num_heads 8 \
  --d_ff 1022 --dropout 0.1 --max_src_len 50 --max_tgt_len 50 \
  --strategy beam_search --beam_width 5 ./data/train.json ./data/dev.json ./data/test.json
