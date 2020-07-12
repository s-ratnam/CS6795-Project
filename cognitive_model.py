import pandas as pd
import numpy as np


def process_tool_files(tool_file, debug=False):
    """
    This function processes the excel sheet containing different tools' features

    :param tool_file:
    :param debug:
    :return:
    """

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

    simple_tool_feats = np.delete(tool_feats, hard_columns, axis=1)
    simple_feat_names = np.array(feat_names)
    simple_feat_names = list(np.delete(simple_feat_names, hard_columns))
    simple_tool_feats = simple_tool_feats.astype(np.int)

    if debug:
        print(tool_names)
        print(feat_names)
        print(tool_feats)
        print(simple_tool_feats)
        print(simple_feat_names)

    return tool_names, simple_feat_names, simple_tool_feats


def process_requirement_file(requirement_file, debug=False):
    """
    This function processes the excel sheet containing mappings from different meeting requirements to different tool
    features

    :param requirement_file:
    :param debug:
    :return:
    """
    pd.read_excel(requirement_file)
    df = pd.read_excel(requirement_file)
    feat_names = list(df.columns)[1:]
    req_feats = df.to_numpy()
    req_names = list(req_feats[:, 0])
    req_feats = req_feats[:, 1:]
    num_reqs, num_feats = req_feats.shape

    hard_columns = set()
    for row in range(num_reqs):
        for col in range(num_feats):
            if isinstance(req_feats[row][col], str):
                if "Yes" in req_feats[row][col]:
                    req_feats[row][col] = 1
                elif "No" in req_feats[row][col]:
                    req_feats[row][col] = 0
                else:
                    hard_columns.add(col)
            elif np.isnan(req_feats[row][col]):
                req_feats[row][col] = 0
    hard_columns = list(hard_columns)

    simple_req_feats = np.delete(req_feats, hard_columns, axis=1)
    simple_feat_names = np.array(feat_names)
    simple_feat_names = list(np.delete(simple_feat_names, hard_columns))
    simple_req_feats = simple_req_feats.astype(np.int)

    if debug:
        print(req_names)
        print(feat_names)
        print(req_feats)
        print(simple_req_feats)
        print(simple_feat_names)

    return req_names, simple_feat_names, simple_req_feats


def recommend_tools(tool_names, feat_names, tool_feats, desired_feats):
    available_req_feats = []
    available_req_feat_names = []
    for rf in desired_feats:
        if rf in feat_names:
            available_req_feats.append(feat_names.index(rf))
            available_req_feat_names.append(rf)

    available_req_feats = np.array(available_req_feats)

    scores = np.sum(tool_feats[:, available_req_feats], axis=1)

    rankings = np.argsort(scores)[::-1]
    # ToDo: the top 5 may not all have score > 0
    top5 = rankings[:5]
    top5_names = list(np.array(tool_names)[top5])
    print("*" * 100)
    print("Given the following desired features: {}".format(", ".join(desired_feats)))
    print("The top 5 recommended tools are {}".format(", ".join(top5_names)))
    for tool in top5_names:
        support_feats = np.array(available_req_feat_names)[tool_feats[tool_names.index(tool), available_req_feats] == 1]
        print("{} supports {}".format(tool, ", ".join(support_feats)))

    return top5_names


def get_desired_features(req_names, feat_names, req_feats, requirements):
    # filter requirements that are not in the list
    available_reqs = []
    available_req_names = []
    for r in requirements:
        if r in req_names:
            available_reqs.append(req_names.index(r))
            available_req_names.append(r)
    available_reqs = np.array(available_reqs)

    # score features based on requirements
    scores = np.sum(req_feats[available_reqs, :], axis=0)
    # rank desired features based on scores
    rankings = np.argsort(scores)[::-1]
    # remove features that have scores not larger than 0
    rankings = rankings[:sum(scores > 0)]
    des_feat_names = list(np.array(feat_names)[rankings])

    print("*" * 100)
    print("Given the following meeting requirements: {}".format(", ".join(requirements)))
    print("The desired tool features are {}".format(", ".join(des_feat_names)))
    for req in available_req_names:
        des_feats = np.array(feat_names)[req_feats[req_names.index(req)] == 1]
        print("{} requires features: {}".format(req, ", ".join(des_feats)))

    return des_feat_names


def run():
    tool_file = "data/tools.xlsx"
    requirement_file = "data/requirements.xlsx"
    tool_names, feat_names, tool_feats = process_tool_files(tool_file, debug=False)
    req_names, feat_names_copy, req_feats = process_requirement_file(requirement_file, debug=False)

    assert feat_names_copy == feat_names, "tools.xlsx and requirements.xlsx have different tool features"

    reqs = list(np.random.choice(req_names, 2, replace=False))
    des_feat_names = get_desired_features(req_names, feat_names, req_feats, reqs)
    recommend_tools(tool_names, feat_names, tool_feats, des_feat_names)

    for i in range(10):
        print("\n\n")
        reqs = list(np.random.choice(req_names, 2, replace=False))
        des_feat_names = get_desired_features(req_names, feat_names, req_feats, reqs)
        recommend_tools(tool_names, feat_names, tool_feats, des_feat_names)


if __name__ == "__main__":
    run()