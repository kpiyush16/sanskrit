# Sanskrit Word Segmentation by CopyNet + nmt (tensorflow)

Requirements:
* Python 2.7.
* subword-nmt
```bash
pip install subword-nmt
```
* tensorflow > 1.5.0 (consider to use the latest [Ana-]conda release)

## Instruction to generate Byte Pair Encoded (BPE) data
* Change directory to nmt/BPE_2ch
* Unzip data_split.tar.gz
* run the "inps.sh" file
```bash
./inps.sh
```

## Testing while Training
```bash
python2.7 -m nmt.nmt --copynet=True --share_vocab=True --attention=scaled_luong --src=src --tgt=trg --vocab_prefix=nmt/BPE_2ch/vocab --train_prefix=nmt/BPE_2ch/train_BPE --test_prefix=nmt/BPE_2ch/test_BPE --dev_prefix=nmt/BPE_2ch/valid_BPE --out_dir=nmt/models_2ch/ --num_train_steps=80000 --steps_per_stats=100 --encoder_type=bi --num_layers=4 --num_units=128 --dropout=0.4 --metrics=bleu --num_keep_ckpts=80 --decay_scheme=luong10 --batch_size=64 --src_max_len=150 --tgt_max_len=150
```

## Only Training
```bash
python2.7 -m nmt.nmt --copynet=True --share_vocab=True --attention=scaled_luong --src=src --tgt=trg --vocab_prefix=nmt/BPE_2ch/vocab --train_prefix=nmt/BPE_2ch/train_BPE --dev_prefix=nmt/BPE_2ch/valid_BPE --out_dir=nmt/models_2ch/ --num_train_steps=80000 --steps_per_stats=100 --encoder_type=bi --num_layers=4 --num_units=128 --dropout=0.4 --metrics=bleu --num_keep_ckpts=80 --decay_scheme=luong10 --batch_size=64 --src_max_len=150 --tgt_max_len=150
```
## Only Testing
```bash
python2.7 -m nmt.nmt --out_dir=nmt/models_2ch --inference_input_file=nmt/infer_file.L1 --inference_output_file=nmt/models_2ch/output_infer
```

## Vocabulary Setting

Since in copynet scenarios the target sequence contains words from source sentences, the best choice is to use a **shared vocabulary** for source vocabulary and target vcabulary. And we also use a parameter **generated  vocabulary size**, namely, the number of target vocabulary excluding  words from source sequences (copied words), to indicate that the first N(=generated vocabulary size) words in shared vocabulary are in generate mode and target word indexes larger than N are copied.

In this codebase, `vocab_size` and `gen_vocab_size` are variables representing shared vocabulary size and generated vocabulalry size.
