
```bash
Code will be released upon acceptance.
```

### Training Scripts
```bash
 python -m experiments one train yourdataset_default --gpu_id yourid
```

### test Scripts
```bash
 python -m experiments one test fake/simple_swapping/simple_interpolation --gpu yourid --resume_iter youriter
```

## Acknowledgment
This code base heavily relies on [pytorch implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) and [SA](https://github.com/taesungp/swapping-autoencoder-pytorch). 