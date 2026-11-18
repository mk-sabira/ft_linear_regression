import csv
import json
import sys


def load_data(filename):
    """Load km and price columns from a CSV file."""

    mileages = []
    prices = []

    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: cannot read dataset ({e})")
        sys.exit(1)
    if not mileages:
        print("Error: dataset is empty")
        sys.exit(1)
    return mileages, prices


def normalize_data(data):
    """Normalizing an entire list of data (Min-Max Scaling)"""
    min_val = min(data)
    max_val = max(data)

    span = max_val - min_val
    if span == 0:
        return [0.0] * len(data), min_val, max_val

    scaled_data = [(x - min_val) / span for x in data]

    return scaled_data, min_val, max_val


def train(norm_mileages, prices, learning_rate=0.1, epochs=10000):
    """Batch Gradient Descent training loop"""

    m = len(norm_mileages)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(epochs):
        errors = [
            (theta0 + theta1 * norm_mileages[i]) - prices[i]
            for i in range(m)
        ]

        tmp0 = learning_rate * (1 / m) * sum(errors)
        tmp1 = learning_rate * (1 / m) * sum(
            errors[i] * norm_mileages[i] for i in range(m))
        
        theta0 -= tmp0
        theta1 -= tmp1

    return theta0, theta1


def denormalize_data(theta0, theta1, min_km, max_km):
    """Convert thetas from the normalized scale to the raw km scale."""

    span = max_km - min_km
    if span == 0:
        return theta0, 0.0
    real_theta1 = theta1 / span
    real_theta0 = theta0 - real_theta1 * min_km
    return real_theta0, real_theta1


def main():
    mileages, prices = load_data("data.csv")

    norm_mileages, min_km, max_km = normalize_data(mileages)

    theta0, theta1 = train(norm_mileages, prices)
    theta0, theta1 = denormalize_data(theta0, theta1, min_km, max_km)

    print("Training complete!")
    print(f"Trained Parameters: theta0 = {theta0}, theta1 = {theta1}")

    output_data = {
        "theta0": theta0,
        "theta1": theta1
    }

    with open("thetas.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("Parameters saved to thetas.json")


if __name__ == "__main__":
    main()
