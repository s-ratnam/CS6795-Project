import pandas as pd
import numpy as np


def process_tool_files(tool_file):

    pd.read_excel(tool_file)
    df = pd.read_excel(tool_file)
    feat_names = list(df.columns)[1:]
    tool_feats = df.to_numpy()
    tool_names = list(tool_feats[:, 0])
    tool_feats = tool_feats[:, 1:]
    num_tools, num_feats = tool_feats.shape
    hard_columns = set()
    for row in range(num_tools):
        for col in range(num_feats):
            if isinstance(tool_feats[row][col], str):
                if "Yes" in tool_feats[row][col]:
                    tool_feats[row][col] = 1
                elif "No" in tool_feats[row][col]:
                    tool_feats[row][col] = 0
                else:
                    hard_columns.add(col)
            elif np.isnan(tool_feats[row][col]):
                tool_feats[row][col] = 0
    hard_columns = list(hard_columns)

    # print(tool_names)
    # print(feat_names)
    # print(tool_feats)

    simple_tool_feats = np.delete(tool_feats, hard_columns, axis=1)
    simple_feat_names = np.array(feat_names)
    simple_feat_names = list(np.delete(simple_feat_names, hard_columns))
    simple_tool_feats = simple_tool_feats.astype(np.int)
    # print(simple_tool_feats)
    # print(simple_feat_names)

    return tool_names, simple_feat_names, simple_tool_feats


def predict_tools(tool_names, feat_names, tool_feats, req_feats):
    available_req_feats = []
    available_req_feat_names = []
    for rf in req_feats:
        if rf in feat_names:
            available_req_feats.append(feat_names.index(rf))
            available_req_feat_names.append(rf)

    available_req_feats = np.array(available_req_feats)

    scores = np.sum(tool_feats[:, available_req_feats], axis=1)

    rankings = np.argsort(scores)[::-1]
    top5 = rankings[:5]
    top5_names = list(np.array(tool_names)[top5])
    print("*"*100)
    print("Given the following desired features: {}".format(", ".join(req_feats)))
    print("The top 5 recommended tools are {}".format(", ".join(top5_names)))
    for tool in top5_names:
        support_feats = np.array(available_req_feat_names)[tool_feats[tool_names.index(tool), available_req_feats] == 1]
        print("{} supports {}".format(tool, ", ".join(support_feats)))


def run():
    tool_file = "data/tools.xlsx"
    tool_names, feat_names, tool_feats = process_tool_files(tool_file)

    for i in range(10):
        req_feats = list(np.random.choice(feat_names, 4, replace=False))
        predict_tools(tool_names, feat_names, tool_feats, req_feats)









if __name__ == "__main__":
    run()