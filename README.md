# HGT
Recent advances in heterophily and graph diffusion models have attracted considerable attention, but the capabilities  of graph diffusion have not been incorporated into the heterophily network. Traditional graph diffusion models struggle with heterophily because modelling unique diffusion styles for heterophily data in graph diffusion is difficult, particularly in terms of information propagation and the incorporation of local context. In this paper, we propose the first model to combine graph diffusion with heterophily, Heterophily Graph Diffusion (HGD), which integrates local information to better capture the diversity and complexity of heterophily data. Using pseudo-labels, we introduce an attention mechanism that learns the significance of local information, translating this into a diffusivity matrix that guides the diffusion process. Our model outperforms the best available graph diffusion model by 0.09-0.11  in heterophily data, and is comparable to the sota of existing graph heterophily models. This work offers new insights into the development of graph diffusion models, highlighting their potential for heterophily data applications.

before run XXX.sh, change the file_path in code first.

Beside run XXX.sh, you can also run python GCN-test.py --run_num 2 --k 2 --dataset chameleon-filtered --selfMask true --train_ratio 0.65 --val_ratio 0.05 --otherEmbeding True --typeEmbeding HGT --diffusion True --alpha 0.5 --beta 1 --gamma 0.5 --device cuda:1
in your PYcharm.
More about environment in requirement.txt

