# Continuous missing values imputation using Autoencoders

[![PyPI version](https://badge.fury.io/py/autoencoder-impute.svg)](https://badge.fury.io/py/autoencoder-impute)
[![Downloads](https://pepy.tech/badge/autoencoder-impute)](https://pepy.tech/project/autoencoder-impute)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of imputation using different types of autoencoders described in [paper](https://google.com). This 
python package contains implementation of Denoising Autoencoder ([autoencoder_impute.DaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L199)),
Multimodal Autoencoder ([autoencoder_impute.MaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L355)),
Variational Autoencoder ([autoencoder_impute.VaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L494))

## Installation

```bash
pip install autoencoder-impute
```

## Usage

### Denoising autoencoder (DAE)

![Denoising autoencoder (DAE)](https://raw.githubusercontent.com/mbalatsko/autoencoder-impute/master/img/overcomplete-2-2/overcomplete-2-2-1.png)

Denoising Autoencoder ([autoencoder_impute.DaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L199))

```python
from autoencoder_impute import DaeImpute

# Initialization
dae = DaeImpute(k=5, h=7)

# Training
dae.fit(df_train, df_val)

# Imputation
reconstructed_df = dae.transform(df_missing)

# Raw DAE output
raw_dae_out_df = dae.predict(df_missing)

# Save model
dae.save('dataset-k-5-h-5', path='.')

# Load model
new_dae = DAE(k=5, h=7)
new_dae.load('dataset-k-5-h-5', path='.')
```

### Multimodal autoencoder (MAE)

![Multimodal autoencoder (MAE)](https://raw.githubusercontent.com/mbalatsko/autoencoder-impute/master/img/mae/mae-1.png)

Multimodal Autoencoder ([autoencoder_impute.MaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L355))

```python
from autoencoder_impute import MaeImpute

# Initialization
mae = MaeImpute(k=5, h=7)

# Training
mae.fit(df_train, df_val)

# Imputation
reconstructed_df = mae.transform(df_missing)

# Raw DAE output
raw_dae_out_df = mae.predict(df_missing)

# Save model
mae.save('dataset-k-5-h-5', path='.')

# Load model
new_dae = MAE(k=5, h=7)
new_dae.load('dataset-k-5-h-5', path='.')
```

### Variational autoencoder (VAE)

![Variational autoencoder (VAE)](https://raw.githubusercontent.com/mbalatsko/autoencoder-impute/master/img/vae/vae-1.png)

Variational Autoencoder ([autoencoder_impute.VaeImpute](https://github.com/mbalatsko/autoencoder-impute/blob/master/autoencoder_impute/models.py#L494))

```python
from autoencoder_impute import VaeImpute

# Initialization
vae = VaeImpute(k=5, h=7)

# Training
vae.fit(df_train, df_val)

# Imputation
reconstructed_df = vae.transform(df_missing)

# Raw DAE output
raw_dae_out_df = vae.predict(df_missing)

# Save model
vae.save('dataset-k-5-h-5', path='.')

# Load model
new_dae = VAE(k=5, h=7)
new_dae.load('dataset-k-5-h-5', path='.')
```
