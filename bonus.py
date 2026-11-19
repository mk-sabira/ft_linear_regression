import math
import matplotlib.pyplot as plt
from train import load_data
from predict import load_thetas, estimate_price


def plotting(mileages, prices, theta0, theta1):
    """Plot data points and regression line."""
    plt.scatter(mileages, prices, color='blue', label='Data')

    x_line = [min(mileages), max(mileages)]
    y_line = [estimate_price(x, theta0, theta1) for x in x_line]

    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.scatter(x_line, y_line, color='green', label='Line endpoints')

    plt.title("Linear regression")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.png")


def mse(y_real, y_pred):
    """Mean Squared Error: the average of the squared errors.

    Squaring stops positive and negative errors from cancelling out
    and punishes big mistakes more. The unit is price squared."""

    m = len(y_real)
    return (1 / m) * sum((y_pred[i] - y_real[i]) ** 2 for i in range(m))


def rmse(y_real, y_pred):
    """Root Mean Squared Error: the square root of the MSE.

    It is back in the same unit as the price, so it reads as
    "the typical error of a prediction"."""

    return math.sqrt(mse(y_real, y_pred))


def mae(y_real, y_pred):
    """Mean Absolute Error: the average of the absolute errors.

    It is in price units and easy to read. Compared to RMSE, it is
    less sensitive to a few very large errors (outliers).
    """

    m = len(y_real)
    return (1 / m) * sum(abs(y_real[i] - y_pred[i]) for i in range(m))


def r2(y_real, y_pred):
    """Coefficient of determination (R squared).

    R2 = 1 - ss_res / ss_tot. It is the share of the price variation
    explained by the model: 1 is perfect, 0 is no better than always
    predicting the mean price, and below 0 is worse than that.
    """

    m = len(y_real)

    # ss_res: residual sum of squares, the error left by our model
    ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(m))

    # ss_tot: total sum of squares, the error of a baseline model that
    # always predicts the mean price (the total variance in the data)

    mean = sum(y_real) / m
    ss_tot = sum((y_real[i] - mean) ** 2 for i in range(m))
    # All prices identical: no variance to explain, R2 is undefined.
    # We return 0.0 by convention to avoid dividing by zero.
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def main():
    """Load the data and trained thetas, then plot and
    print the model's metrics."""

    mileages, prices = load_data("data.csv")
    theta0, theta1 = load_thetas()

    plotting(mileages, prices, theta0, theta1)

    predictions = [estimate_price(x, theta0, theta1) for x in mileages]

    print(f"MSE:  {mse(prices, predictions):.2f} (average squared error)")
    print(
        f"RMSE: {rmse(prices, predictions):.2f} "
        "(typical error, in price units)"
    )
    print(
        f"MAE:  {mae(prices, predictions):.2f} "
        "(average absolute error)"
    )
    print(
        f"R2:   {r2(prices, predictions):.4f} "
        f" or {r2(prices, predictions) * 100:.2f} percent of accuracy"
    )


if __name__ == "__main__":
    main()
