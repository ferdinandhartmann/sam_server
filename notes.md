https://qiita.com/takusandayo/items/55ed52f9298064d7cf61

pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

python -m pip install fast-simplification

export PYTHONNOUSERSITE=1
alias pip='python -m pip'

python -m sam_server.scripts.sam_3d_worker
