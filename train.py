import csv


def load_data(filename):
    """Loading data function"""

    mileages = []
    prices = []

    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            print(type(reader))
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))
    except FileNotFoundError:
        print("Error: Dataser file not found")
    return mileages, prices


def normalize_data(data):
    """Normalizing an entire list of data (Min-Max Scaling)"""
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return [0.0 for _ in data]
    
    return [(x - min_val) / (max_val - min_val) for x in data]


def main():
    mileages, prices = load_data("data.csv")
    # print(prices)


    norm_mileages = normalize_data(mileages)
    norm_prices = normalize_data(prices)


    # print(norm_prices)





if __name__ == "__main__":
    main()