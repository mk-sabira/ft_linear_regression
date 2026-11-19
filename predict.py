import json
import os


def load_thetas(filename="thetas.json"):
    """Loads theta0 and theta1 from a JSON file. Defaults
to 0 if file doesn't exist."""

    if not os.path.exists(filename):
        return 0.0, 0.0
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return (float(data.get("theta0", 0.0)),
                    float(data.get("theta1", 0.0)))
    except Exception:
        return 0.0, 0.0


def estimate_price(mileage, theta0, theta1):
    """Calculates the estimated price using the linear hypothesis formula."""
    return theta0 + (theta1 * mileage)


def main():
    """Load trained parameters, prompt for car mileage,
and print the estimated price."""

    theta0, theta1 = load_thetas()

    print(f"Current model parameters loaded -> theta0: \
{theta0}, theta1: {theta1}")

    try:
        user_input = input("Please enter the car mileage (km): ")
        mileage = float(user_input)

        if mileage < 0:
            print("Error: Mileage cannot be negative.")
            return

        price = estimate_price(mileage, theta0, theta1)

        print(f"Estimated price for {mileage} km: {price:.2f}")

    except ValueError:
        print("Error: Please enter a valid number for mileage.")


if __name__ == "__main__":
    main()
