import os
import json


def concat_data():
    data = []
    for file in os.listdir("./data"):
        try:
            if "docs_with" in file or '.DS_Store' in file:
                continue
            with open(f"./data/{file}", "r") as f:
                d_ = json.load(f)
                data.append({"messages": d_["obj"]})
        except:
            ...
    with open("./result/composed_data.json", "w") as f:
        json.dump(data, f, indent=4)
    return data


if __name__ == "__main__":
    concat_data()
