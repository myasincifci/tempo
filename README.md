## Training tempo model and evaluate:
```
python tools/train_net_new.py \
    --eval linear \
    --epochs 60 \
    --proximity 30 \
    --save_model asl_big_e10_p30_run5.pth
```

## Linear Evaluation:
```
python tools/linear_eval.py \
    --path model_zoo/asl_big_e10_p30_run5.pth
```

## Semi-Supervised Evaluation:
```
python tools/semi_sup_eval.py \
    --path model_zoo/asl_big_e10_p30_run5.pth
```