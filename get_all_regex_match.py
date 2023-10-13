import os
from time import sleep
from typing import List, Union
import numpy as np
import json
from lean_dojo import LeanGitRepo, trace

def main(file_name: str, regex: str, max_sample_size=-1, seed=69_420):
  """
  Set max_sample_size to -1 to have no limit
  """
  if max_sample_size != -1:
    raise NotImplementedError("max_sample_size != -1 not implemented")

  np.random.seed(seed)
  # TODO: figure the below out...
  repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "42b62d41dd723321a3a0801f4d9e583625f90311")
  dst_dir = "traced_mathlib4"
  # repo = LeanGitRepo("https://github.com/Lev-Stambler/mathlib-clustering", "6069641a14213014f6e92fad3280e3a5f524497c")
  file_path = f"data_store/" + file_name + ".json"

  repo = trace(repo, dst_dir=dst_dir)
  thms = repo.get_traced_theoraems()

  rets = []

  for i, thm in enumerate(thms):
    print(thm)
    rets += [thm]

  json.dump(rets, open(file_path, "w"))
  # TODO: write to JSON

if __name__ == "__main__":
  main("all_theorems", ".*", max_sample_size=-1, seed=69_420)