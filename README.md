We are continuously extracting and updating code from our projects, and the optimization process will continue and take some time.

# train & test model
run train_ASTGCN.py

# hyper parameters setting
1. set embedding dimension ($ed$) from {10,20,30,40,50,60,70,80,90,100,110,120} in model/ASTGCN_r.py
2. set filtration factor ($/theta$) from {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9} in model/ASTGCN_r.py
3. set input length ($il$) from {2,5,10,15,20,25,30} in configurations/ACL18_aastgcn.conf

Please refer to our paper for seeking the best parameter sets and fine-tuning the best prediction performance.