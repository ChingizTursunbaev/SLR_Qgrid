import numpy as np

gloss_dict = np.load("/home/kks/workspace/slr/data/phoenix2014/gloss_dict.npy", allow_pickle=True).item()  # {str: [idx, ...]}

print(gloss_dict)