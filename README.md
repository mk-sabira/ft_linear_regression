# ft_linear_regression

An introduction to machine learning: predict the price of a car from its
mileage with a linear function trained by gradient descent.

Model: `estimatePrice(mileage) = theta0 + theta1 * mileage`

## Files

| File | Purpose |
|---|---|
| `train.py` | Reads `data.csv`, trains the model, saves `thetas.json` |
| `predict.py` | Asks for a mileage and prints the estimated price |
| `bonus.py` | Plots data + regression line (`plot.png`) and prints metrics |
| `data.csv` | Dataset (`km`, `price`) |

## Requirements

- Python 3
- `train.py` and `predict.py` use only the standard library
  (`csv`, `json`, `os`, `sys`).
- The bonus needs `matplotlib`.

## Usage

```bash
python3 predict.py   # before training: theta0 = theta1 = 0, price is 0
python3 train.py     # trains and saves thetas.json
python3 predict.py   # enter a mileage, e.g. 100000
python3 bonus.py     # saves plot.png and prints MSE, RMSE, MAE, R2
```

## Virtual environment (only if matplotlib is not found)

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -c "import matplotlib; print(matplotlib.__version__)"
```

`--system-site-packages` reuses the matplotlib installed on the machine,
so nothing is downloaded. To remove it: `deactivate && rm -rf venv`.

## How it works

1. Mileages are min-max normalized to [0, 1] so one learning rate suits
   both parameters.
2. Batch gradient descent, with a simultaneous update of both thetas:

   ```
   tmp0 = lr * (1/m) * sum(estimatePrice(x[i]) - price[i])
   tmp1 = lr * (1/m) * sum((estimatePrice(x[i]) - price[i]) * x[i])
   ```

3. The thetas are converted back to the raw km scale before saving, so
   `predict.py` uses the plain hypothesis `theta0 + theta1 * mileage`.

## Bonus: metrics (computed on the training data)

- **MSE**: average squared error.
- **RMSE**: square root of MSE, in price units.
- **MAE**: average absolute error, in price units.
- **R2**: `1 - SS_res / SS_tot`. 1 is perfect, 0 is no better than
  predicting the mean price, below 0 is worse.

Classification metrics (accuracy, precision, recall, F1) do not apply,
since the model predicts a continuous value.
