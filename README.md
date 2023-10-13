For some reason, the docker does not seem to work for LeanDojo.
We can instead set the CONTAINER env variable to `native`.
We thus run 
```
CONTAINER=native python3 <file using leandojo>.py
```