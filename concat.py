import os
import json
import numpy as np


def concat_data(eval_ratio: float):
    data = []
    nn = 0
    import jsonlines
    for file in os.listdir("./data"):
        try:
            if "docs_with" in file or '.DS_Store' in file or "FF_" not in file and "AA_" not in file:
                continue
            with open(f"./data/{file}", "r") as f:
                d_ = json.load(f)

                nn = max(nn, len(d_["obj"][0]["content"].split(" ")))
                print(d_["obj"][0]["content"].split(" "))

                data.append({"messages": d_["obj"]})
        except:
            ...

    N = len(data)
    train_N = int(N * (1 - eval_ratio))
    np.random.seed(42)
    np.random.shuffle(data)
    train_data =  data[:train_N]
    eval_data = data[train_N:]

    with open("./result/train_data.json", "w") as f:
        writer = jsonlines.Writer(f)
        writer.write_all(train_data)
    with open("./result/eval_data.json", "w") as f:
        writer = jsonlines.Writer(f)
        writer.write_all(eval_data)

    print(f"nn: {nn}")


if __name__ == "__main__":
    concat_data(eval_ratio=.1)
