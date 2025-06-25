```angular2html
python GCN-test.py --dataset chameleon    --otherEmbeding True  --typeEmbeding HGT  --diffusion True  --device cuda:1
```

```angular2html
python GCN-test.py --dataset chameleon --conv True    --otherEmbeding True  --typeEmbeding HGT  --diffusion True  --device cuda:1
```

```angular2html
python GCN-test.py --dataset squirrel    --otherEmbeding True  --typeEmbeding HGT  --diffusion True  --device cuda:1
```

```angular2html 
python GCN-test.py --run_num 2 --dataset squirrel  --otherEmbeding True  --typeEmbeding HGT   --device cuda:1 --k 2
```

```angular2html
python GCN-test.py --run_num 2 --dataset chameleon --conv True --diffusion True  --otherEmbeding True  --typeEmbeding HGT   --device cuda:1 --k 2
```

```angular2html
python GCN-test.py --run_num 1 --dataset PubMed --conv true --diffusion True --otherEmbeding True  --typeEmbeding HGT  --device cuda:0 --alpha 0.5 --beta 1 --gamma 0 --k 2
```

```angular2html
python GCN-test.py --run_num 1 --dataset CiteSeer --conv true   --device cuda:0 --alpha 0.5 --beta 1 --gamma 0.2 --k 2
```

```angular2html
python GCN-test.py --run_num 1 --dataset CiteSeer  --diffusion True  --device cuda:0 --k 2
```

```angular2html
python GCN-test.py --run_num 1 --dataset CiteSeer --selfMask true  --diffusion True  --device cuda:0 --k 2
```

```angular2html
python GCN-test.py --run_num 5 --k 2 --dataset chameleon --conv True --selfMask true --train_ratio 0.85  --val_ratio 0.05    --otherEmbeding True  --typeEmbeding HGT  --diffusion True  --device cuda:1
```


```angular2html
python GCN-test.py --run_num 2 --k 2 --dataset PubMed  --selfMask true --train_ratio 0.05  --val_ratio 0.45    --otherEmbeding True  --typeEmbeding HGT  --diffusion True --alpha 0.5 --beta 1 --gamma 0.2  --device cuda:1
python GCN-test.py --run_num 1 --dataset PubMed --selfMask true --train_ratio 0.05  --val_ratio 0.45  --conv True  --device cuda:0 --k 2
python GCN-test.py --run_num 1 --dataset PubMed --selfMask true --train_ratio 0.05  --val_ratio 0.45  --diffusion True --device cuda:0 --k 2
```


```angular2html
python GCN-test.py --run_num 2 --k 2 --dataset amazon-ratings  --selfMask true --train_ratio 0.45  --val_ratio 0.05    --otherEmbeding True  --typeEmbeding HGT  --diffusion True --alpha 0.5 --beta 1 --gamma 0.2  --device cuda:1
python GCN-test.py --run_num 1 --dataset PubMed --selfMask true --train_ratio 0.05  --val_ratio 0.45  --conv True  --device cuda:0 --k 2
python GCN-test.py --run_num 1 --dataset PubMed --selfMask true --train_ratio 0.05  --val_ratio 0.45  --diffusion True --device cuda:0 --k 2
python GCN-test.py --run_num 2 --k 4 --dataset squirrel-filtered --selfMask true --train_ratio 0.55 --val_ratio 0.05 --otherEmbeding True --typeEmbeding HGT --diffusion True --alpha 0.4 --beta 1 --gamma 0.4 --device cuda:1
python GCN-test.py --run_num 2 --k 2 --dataset chameleon-filtered --selfMask true --train_ratio 0.65 --val_ratio 0.05 --otherEmbeding True --typeEmbeding HGT --diffusion True --alpha 0.5 --beta 1 --gamma 0.5 --device cuda:1

```