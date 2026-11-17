from train import load_data
from predict import load_thetas, estimate_price
import matplotlib.pyplot as plt


def main():
    mileages, prices = load_data("data.csv")

    theta0, theta1 = load_thetas()

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


if __name__ == "__main__":
    main()
