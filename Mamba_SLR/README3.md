### Step 1
```bash
pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  torchaudio==2.4.0+cu121 \
  torch-tb-profiler==0.4.3 \
  torchmetrics==1.7.4 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 2
For making system use the toolkit from your conda environment rather than the **system's** one (which might be old to build the `causal-conv1d` package.)
```bash
# inside your (ch_slr) env:
conda install nvidia \
   cuda-toolkit=12.1 \
   cuda-nvcc=12.1

# since the nvcc looks at CUDA_HOME so the CUDA wheels installed 
# in the conda env should be pointed to by the CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX

# verify
nvcc -V       # should now report 12.1.x

# now install everything (including causal-conv1d)
pip install -r pip_list.txt
```

### Step 3
```bash
# apex package installation
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install --no-build-isolation \
--verbose --no-cache-dir \ 
--global-option="--cpp_ext" \ 
--global_option="--cuda_ext" ./
cd ..
rm -rf apex

# Sometimes mamba-ssm=2.2.4 also gives error. Use your 
# environments installed dependency, while installing this one 
# from the pip with `--no-build-isolation` flag, instead of the 
# pip's default option of installing the package in isolation in 
# separate env
pip install --no-build-isolation mamba-ssm==2.2.4
```
