# Examples

`train_example.py` is a pure-Python stand-in for a training script. It prints a few progress lines and writes `outputs/result.txt` in the current working directory. It does not import this package and does not call Kaggle, Colab, or the network.

```bash
cd examples
python train_example.py
```

`sdk_example.py` loads that script through `Client` in mock mode. It recommends the offline fake provider, runs the script, and prints the job id. Quota text in the output is a test fixture. State stays in a temporary directory.

`config.multi-account.yaml` is a mock-mode config with credential references only. It does not contain tokens. Account selection does not rotate when a quota is exhausted. See `docs/accounts.md`.

```bash
python examples/sdk_example.py
```
