import pandas as pd
import sys

def median(t):
    n = len(t)
    if (n - 1) % 2 == 0:
        return t[n // 2]
    return (t[n//2 - 1] + t[n//2]) / 2

def first_quartile(t) -> float:
    n = len(t)
    kind = (n-1) % 4
    return (t[(n-1) // 4] * (4-kind) + (t[(n-1) // 4 + 1] * kind if kind else 0)) / 4

def third_quartile(t) -> float:
    return first_quartile(t[::-1])

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Please provide filename.")
        exit()

    filename = sys.argv[1]
    try:
        file = pd.read_csv(filename).select_dtypes(include='number')
    except:
        print(f"File '{filename}' does not exist or cannot be read.")
        exit(1)


    mydescribe = {
        "Features" : [],
        "Count" : [],
        "Mean" : [],
        "Std" : [],
        "Min" : [],
        "25%" : [],
        "50%" : [],
        "75%" : [],
        "Max" : []
    }

    for feature in file:
        mydescribe["Features"].append( feature )
        serie = []
        for f in file[feature]:
            if not pd.isna(f):
                serie.append(f)
        serie.sort()

        mydescribe["Count"].append( len(serie) )
        mean = sum(serie) / len(serie)
        mydescribe["Mean"].append( mean )
        mydescribe["Std"].append(sum((ti - mean)**2 / (len(serie)-1) for ti in serie)**0.5) # f...
        mydescribe["Min"].append(serie[0])
        mydescribe["25%"].append(first_quartile(serie))
        mydescribe["50%"].append(median(serie))
        mydescribe["75%"].append(third_quartile(serie))
        mydescribe["Max"].append(serie[-1])

    for k in mydescribe:
        print("{:7s}".format(k if k != "Features" else ""), end="")
        for v in mydescribe[k]:
            if k == "Features":
                print("{:>15.15s}".format(v), end=" ")
            else:
                print("{:15.6f}".format(v), end=" ")
        print()
