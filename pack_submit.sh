# 0. Copy from checkout/
# (biocas) czz@v:~/biocas/biocas-challenge-22/saved$ tree .
# .
# ├── model11
# │   ├── config.json
# │   └── model_best11.pth
# ├── model12
# │   ├── config.json
# │   └── model_best12.pth
# ├── model21
# │   ├── config.json
# │   └── model_best21.pth
# └── model22
#     ├── config.json
#     └── model_best22.pth


# 1. Sync from vector compute
rsync -avzh czz@v.vectorinstitute.ai:/h/czz/biocas/biocas-challenge-22/saved/ ./saved_v/

# 2. Test run
exec 'bash testcase/test.sh'

# 3. Copy documents
# Drop data/SPRSound/clip wav processed
# Drop wandb/
# Drop saved/
