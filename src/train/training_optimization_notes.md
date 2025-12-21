# Notes on optimizing training on hardware

## Background

I was training, but noticed... sub-par performance. The real indicator was that I was only seeing 50% CPU utilization (fixed, which was suspicious) and I didn't seem to be utilizing the integrated GPU.

## DatalLoader Optimization

### Not using N-Workers

`DataLoader` takes a `num_workers` argument, which controls the number of sub-processes preparing to load data. The `DataLoader` class only uses single process by default. Adding this to the `train_loader` and `test_loader` objects slightly less than doubled iterations per second.

Also set `persistent_workers` to `True` to save on overhead of instantiating them epoch-to-epoch.

### GPU Utilization

I'm using a Framework Ryzen 9 AI HX370 motherboard, with an integrated Radeon 890M GPU. `ROCm` installation can be finicky and complex. The tradeoff doesn't seem worth it for a relatively small (~30k params) model.

### Batch Optimization

I had originally had the batch size set at 32. This is good for fine-grained exploration and generalization, but bad for training speed and creates noisery gradients. I didn't want to blow this out and go enormous (where the matrix multiplication efficiency takes big improvements) because there is a tradeoff for converging. 512 is where I've stuck as it seems a good balance anecdotally.

TODO:

- [ ] Create a batch size sweep script to find optimal tradeoff points for batch size and convergence

### Memory Pinning

Set `pin_memory` to `True`. This pre-allocates memory in a way that is faster for transfer.

TODO :

- [ ] Measure the actual impact of this turned on vs turned off
