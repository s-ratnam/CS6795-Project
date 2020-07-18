import pandas as pd
import numpy as np
import pytz
import datetime
from itertools import combinations
import names


# Meta data for requirements
# Participant Category:
# - Hearing Impariment: [True, False]
# - Vision Impariment: [True, False]
# - Introvert: [True, False]
# - Bad Internect Connection: [True, False]
# - Operating System: [MacOS, Windows, Linus]
# - Budget: (0 - inf)
# - Timezone: dropdown list?

# Meeting Category:
# - Time
# - Purpose: [Presentation, Lecture, Chat]
# - Capacity: (2 - inf)

def process_capacity(capacity, verbose=False):
    """
    This function takes in number of participants for the meeting and outputs tools that support it.

    :param capacity:
    :return:
    """
    tool_capacity = {"Microsoft Teams": 250,
                     "Google Meet": 100,
                     "Google Meet G Suite Essential": 150,
                     "Google Meet G Suite Enterprise Essential": 250,
                     "Google Hangouts": 25,
                     "Skype": 20,
                     "Zoom (Paid)": {15: 100, 20: 300, 25: 500},
                     "Zoom": 100,
                     "Cisco WebEx": {13.5: 50, 17.95: 100, 26.95: 200},
                     "BlueJeans": 0,
                     "Slack (paid)": 0,
                     "Slack": 0,
                     "Whatsapp": 3,
                     "Facetime": 32,
                     "HouseParty": 8}

    assert isinstance(capacity, int)

    support_tools = []
    for tool in tool_capacity:
        if isinstance(tool_capacity[tool], int):
            if capacity <= tool_capacity[tool]:
                support_tools.append((tool, 0))
        else:
            for price_tier in tool_capacity[tool]:
                if capacity <= tool_capacity[tool][price_tier]:
                    support_tools.append((tool, price_tier))
    if verbose:
        print("*" * 100)
        print("Given that {} people will be participating in the meeting".format(capacity))
        for tool, price in support_tools:
            if price == 0:
                print("{} will be suitable".format(tool))
            else:
                print("The paid (${}) version of {} will be suitable".format(price, tool))

    return support_tools


def process_systems(tool_names, feat_names, tool_feats, participant_systems, verbose=False):
    """
    This function takes in particpants' operating systems as a list of strings and outputs supported tools

    :param tool_names:
    :param feat_names:
    :param tool_feats:
    :param participant_systems:
    :return:
    """
    tool_systems = {"Microsoft Teams": ["Mac", "Windows", "Linux"],
                     "Google Meet": ["Mac", "Windows", "Linux"],
                     "Google Meet G Suite Essential": ["Mac", "Windows", "Linux"],
                     "Google Meet G Suite Enterprise Essential": ["Mac", "Windows", "Linux"],
                     "Google Hangouts": ["Mac", "Windows", "Linux"],
                     "Skype": ["Mac", "Windows", "Linux"],
                     "Zoom (Paid)": ["Mac", "Windows", "Linux"],
                     "Zoom": ["Mac", "Windows", "Linux"],
                     "Cisco WebEx": ["Mac", "Windows"],
                     "BlueJeans": ["Mac", "Windows", "Linux"],
                     "Slack (paid)": ["Mac", "Windows", "Linux"],
                     "Slack": ["Mac", "Windows", "Linux"],
                     "Whatsapp": ["Mac", "Windows"],
                     "Facetime": ["Mac"],
                     "HouseParty": ["Mac", "Windows"]}

    # check
    assert "Dial In With Phone" in feat_names
    assert "Join from Browser" in feat_names
    for tool in tool_systems:
        assert tool in tool_names

    all_systems = set(participant_systems)
    support_tools = []
    for tool in tool_systems:
        lacking_systems = list(all_systems - set(tool_systems[tool]))
        if len(lacking_systems) == 0:
            support_tools.append((tool, [], []))
        else:
            ti = tool_names.index(tool)
            fi1 = feat_names.index("Dial In With Phone")
            fi2 = feat_names.index("Join from Browser")
            ac_feats = []
            if tool_feats[ti][fi1]:
                ac_feats.append("Dial In With Phone")
            if tool_feats[ti][fi2]:
                ac_feats.append("Join from Browser")
            if ac_feats:
                support_tools.append((tool, lacking_systems, ac_feats))

    if verbose:
        print("*" * 100)
        print("Given that participants are using {}".format(all_systems))
        for tool, lacking_systems, ac_feats in support_tools:
            if not lacking_systems:
                print("{} will be suitable".format(tool))
            else:
                print("{} lacks support for {}. However it has {}".format(tool, ", ".join(lacking_systems), ", ".join(ac_feats)))

    return support_tools


def process_budgets(participant_budgets, verbose=False):
    """
    This function will takes in participants' budgets as a list of numbers and outputs suitable tools

    :return:
    """
    tool_costs = {"Microsoft Teams": 0,
                  "Google Meet": 0,
                  "Google Meet G Suite Essential": 10,
                  "Google Meet G Suite Enterprise Essential": 20,
                  "Google Hangouts": 0,
                  "Skype": 0,
                  "Zoom (Paid)": [15, 20, 25],
                  "Zoom": 0,
                  "Cisco WebEx": [13.5, 17.95, 26.95],
                  "BlueJeans": 0,
                  "Slack (paid)": [10, 20, 30],
                  "Slack": 0,
                  "Whatsapp": 0,
                  "Facetime": 0,
                  "HouseParty": 0}

    for budget in participant_budgets:
        assert budget >= 0

    all_budgets = set(participant_budgets)
    min_budget = min(all_budgets)
    support_tools = []
    for tool in tool_costs:
        if not isinstance(tool_costs[tool], list):
            cost = tool_costs[tool]
            if cost <= min_budget:
                support_tools.append((tool, 0))
        else:
            price_tiers = tool_costs[tool]
            for cost in price_tiers:
                if cost <= min_budget:
                    support_tools.append((tool, cost))

    if verbose:
        print("*" * 100)
        print("Given that participants' budgets range from {} to {}".format(min(all_budgets), max(all_budgets)))
        for tool, price in support_tools:
            if price == 0:
                print("{} will be suitable".format(tool))
            else:
                print("The paid (${}) version of {} will be suitable".format(price, tool))

    return support_tools


def is_meeting_international(participant_timezones, verbose=False):
    """
    This function will takes in a list of timezones for all participants and determines if the meeting is international

    :param participant_timezones:
    :param verbose:
    :return:
    """
    unique_timezones = set(participant_timezones)
    international = False

    # check
    for timezone in unique_timezones:
        assert timezone in pytz.common_timezones, "The input timezone {} is not standard"

    # compute largest time difference
    tzs = [pytz.timezone(tz) for tz in unique_timezones]
    utcnow = pytz.timezone('utc').localize(datetime.datetime.utcnow())  # generic time
    cts = [utcnow.astimezone(tz).replace(tzinfo=None) for tz in tzs]
    largest_diff_hour = 0
    for a, b in combinations(cts, 2):
        diff = abs(a - b).total_seconds() / 3600.0
        if diff > largest_diff_hour:
            largest_diff_hour = diff

    if largest_diff_hour > 0:
        international = True

    if verbose:
        print("*" * 100)
        print("Given participants' timezones: {}".format(unique_timezones))
        print("The largest time difference is {} (hr)".format(largest_diff_hour))
        if international:
            print("The meeting is international")
        else:
            print("The meeting is not international")

    return international


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


def recommend_tools(tool_names, feat_names, tool_feats, desired_feats, available_tools, verbose=False):
    """

    :param tool_names:
    :param feat_names:
    :param tool_feats:
    :param desired_feats:
    :param available_tools: a list of (tool, price)
    :param verbose:
    :return:
    """
    available_req_feats = []
    available_req_feat_names = []
    for rf in desired_feats:
        if rf in feat_names:
            available_req_feats.append(feat_names.index(rf))
            available_req_feat_names.append(rf)

    available_req_feats = np.array(available_req_feats)

    scores = np.sum(tool_feats[:, available_req_feats], axis=1)

    rankings = np.argsort(scores)[::-1]
    rankings = rankings[:sum(scores > 0)]
    desired_tool_names = list(np.array(tool_names)[rankings])
    tool_feature_mappings = {}

    desired_tools = []
    for tool in available_tools:
        if tool[0] in desired_tool_names:
            desired_tools.append(tool)

    if verbose:
        print("*" * 100)
        print("Given the following required features: {}".format(", ".join(desired_feats)))
        if desired_tools:
            print("The recommended tools are {}".format(desired_tools))
            for tool in [t[0] for t in desired_tools]:
                support_feats = np.array(available_req_feat_names)[tool_feats[tool_names.index(tool), available_req_feats] == 1]
                print("{} supports {}".format(tool, ", ".join(support_feats)))
                tool_feature_mappings[tool] = support_feats.tolist()
        else:
            print("No matching tools")

    # print("********************************************************")
    # print("TOOL TO FEATURE MAPPINGS", tool_feature_mappings)
    return [t[0] for t in desired_tools], tool_feature_mappings


def get_required_features(req_names, feat_names, req_feats, requirements, verbose=False):
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
    tool_to_requirement_mapping = {}

    if verbose:
        print("*" * 100)
        print("Given the following meeting requirements: {}".format(", ".join(requirements)))
        print("The required tool features are {}".format(", ".join(des_feat_names)))
        for req in available_req_names:
            des_feats = np.array(feat_names)[req_feats[req_names.index(req)] == 1]
            print("{} requires features: {}".format(req, ", ".join(des_feats)))
            tool_to_requirement_mapping[req] = des_feats.tolist()
    
    return des_feat_names, tool_to_requirement_mapping


def test():
    tool_file = "data/tools.xlsx"
    requirement_file = "data/requirements.xlsx"
    tool_names, feat_names, tool_feats = process_tool_files(tool_file, debug=False)
    req_names, feat_names_copy, req_feats = process_requirement_file(requirement_file, debug=False)

    assert feat_names_copy == feat_names, "tools.xlsx and requirements.xlsx have different tool features"

    # reqs = list(np.random.choice(req_names, 2, replace=False))
    # des_feat_names = get_desired_features(req_names, feat_names, req_feats, reqs)
    # recommend_tools(tool_names, feat_names, tool_feats, des_feat_names)
    #
    # for i in range(10):
    #     print("\n\n")
    #     reqs = list(np.random.choice(req_names, 2, replace=False))
    #     des_feat_names = get_desired_features(req_names, feat_names, req_feats, reqs)
    #     recommend_tools(tool_names, feat_names, tool_feats, des_feat_names)

    participant_systems = []
    for i in range(3):
        system = list(np.random.choice(["Mac", "Windows", "Linux"], 1))[0]
        participant_systems.append(system)
    process_systems(tool_names, feat_names, tool_feats, participant_systems, verbose=True)


def get_requirements(meeting, participants):
    """
    This function maps info about meeting and participants to pre-defined requirements

    :param meeting:
    :param participants:
    :return:
    """
    organizer_name, meeting_purpose, organizer_budget, organizer_system, organizer_timezone, organizer_desired_tool, \
    organizer_desired_features, meeting_hearing_impairment, meeting_vision_impairment = meeting

    binary_requirements = []
    binary_requirements.append("Meeting: {}".format(meeting_purpose))

    timezones = [p[7] for p in participants]
    if is_meeting_international(timezones, verbose=False):
        binary_requirements.append("Meeting: International")

    participant_requirements = []
    participant_systems = []
    participant_budgets = []
    participant_timezones = []
    for participant in participants:
        name, hearing_impairment, vision_impairment, introvert, bad_internet_connection, system, budget, timezone = participant
        if hearing_impairment:
            participant_requirements.append("Participant: Hearing Impairment")
        if vision_impairment:
            participant_requirements.append("Participant: Vision Impairment")
        if introvert:
            participant_requirements.append("Participant: Introvert")
        if bad_internet_connection:
            participant_requirements.append("Participant: Internet Connection")
        participant_systems.append(system)
        participant_budgets.append(budget)
        participant_timezones.append(timezone)

    # incorporate meeting organizer's information
    if meeting_hearing_impairment:
        participant_requirements.append("Participant: Hearing Impairment")
    if meeting_vision_impairment:
        participant_requirements.append("Participant: Vision Impairment")
    participant_systems.append(organizer_system)
    participant_budgets.append(organizer_budget)
    participant_timezones.append(organizer_timezone)

    participant_requirements = list(set(participant_requirements))
    binary_requirements.extend(participant_requirements)
    return binary_requirements, participant_systems, participant_budgets, participant_timezones


def set_meeting(organizer_name, meeting_purpose, organizer_budget, organizer_system, organizer_timezone, organizer_desired_tool, organizer_desired_features,
               meeting_hearing_impairment, meeting_vision_impairment):
    """
    This function checks if the meeting information is in the correct format and combine them

    :param organizer_name:
    :param meeting_purpose:
    :param organizer_budget:
    :param organizer_system:
    :param organizer_timezone:
    :param organizer_desired_tool:
    :param organizer_desired_features:
    :param meeting_hearing_impairment:
    :param meeting_vision_impairment:
    :return:
    """
    assert isinstance(organizer_name, str)
    assert meeting_purpose in ["Presentation", "Lecture", "Chat"]
    assert organizer_budget >= 0
    assert organizer_system in ["Mac", "Windows", "Linux"]
    assert organizer_timezone in pytz.all_timezones
    assert isinstance(organizer_desired_tool, str)
    assert isinstance(organizer_desired_features, list) or isinstance(organizer_desired_features, set)
    for feat in organizer_desired_features:
        assert isinstance(feat, str)
    assert isinstance(meeting_hearing_impairment, bool)
    assert isinstance(meeting_vision_impairment, bool)

    meeting = [organizer_name, meeting_purpose, organizer_budget, organizer_system, organizer_timezone, organizer_desired_tool, organizer_desired_features,
               meeting_hearing_impairment, meeting_vision_impairment]

    return meeting


def create_random_meeting(verbose=False):
    """
    This function randomly creates meeting and organizer information

    :param verbose:
    :return:
    """
    tools = ['Microsoft Teams', 'Google Meet', 'Google Meet G Suite Essential', 'Google Meet G Suite Enterprise Essential',
     'Google Hangouts', 'Skype', 'Zoom (Paid)', 'Zoom', 'Cisco WebEx', 'BlueJeans', 'Slack (paid)', 'Slack', 'Whatsapp',
     'Facetime', 'HouseParty']
    feats = ['Closed Captioning', 'Screen Sharing', 'Mute All', 'Video Off', 'Join from Browser', 'Free', 'Adjustable Layout',
     'Dial In With Phone', 'Raise Hand', 'Chat Messaging', 'Meeting Recording', 'Polling/Surveys',
     'Virtual Background Integration', 'Screen Reader Compatible', '3D Memoji Avatar', 'Live Photo', 'In-chat Games']

    organizer_name = names.get_full_name()
    meeting_purpose = list(np.random.choice(["Presentation", "Lecture", "Chat"], 1))[0]
    organizer_budget = np.random.rand(1)[0] * 100
    organizer_system = list(np.random.choice(["Mac", "Windows", "Linux"], 1))[0]
    organizer_timezone = list(np.random.choice(pytz.common_timezones, 1))[0]
    organizer_desired_tool = list(np.random.choice(tools, 1))[0]
    organizer_desired_features = list(np.random.choice(feats, 3))
    meeting_hearing_impairment = list(np.random.choice([True, False], 1))[0]
    meeting_vision_impairment = list(np.random.choice([True, False], 1))[0]

    if verbose:
        print("#" * 100)
        print("{} is organizing a meeting".format(organizer_name))
        print("Meeting Purpose: {}".format(meeting_purpose))
        print("Organizer's budget: {}".format(organizer_budget))
        print("Organizer's system: {}".format(organizer_system))
        print("Organizer's timezone: {}".format(organizer_timezone))
        print("Organizer's desired tool: {}".format(organizer_desired_tool))
        print("Organizer's desired feats: {}".format(", ".join(organizer_desired_features)))
        if meeting_hearing_impairment:
            print("Organizer thinks that there is participant with hearing impairment")
        if meeting_vision_impairment:
            print("Organizer thinks that there is participant with vision impairment")

    meeting = [organizer_name, meeting_purpose, organizer_budget, organizer_system, organizer_timezone, organizer_desired_tool, organizer_desired_features,
               meeting_hearing_impairment, meeting_vision_impairment]

    return meeting


def create_random_participants(capacity, verbose=False):
    """
    This function randomly creates participants

    :param capacity: the number of participants in the meeting
    :return:
    """
    participants = []
    for i in range(capacity):
        participant_name = names.get_full_name()
        participant_hearing_impairment = list(np.random.choice([True, False], 1))[0]
        participant_vision_impairment = list(np.random.choice([True, False], 1))[0]
        participant_introvert = list(np.random.choice([True, False], 1))[0]
        participant_bad_internet_connection = list(np.random.choice([True, False], 1))[0]
        participant_system = list(np.random.choice(["Mac", "Windows", "Linux"], 1))[0]
        participant_budget = np.random.rand(1)[0] * 100
        participant_timezone = list(np.random.choice(pytz.common_timezones, 1))[0]

        participant = [participant_name, participant_hearing_impairment, participant_vision_impairment, participant_introvert, participant_bad_internet_connection, participant_system,
                       participant_budget, participant_timezone]
        participants.append(participant)

    if verbose:
        print("#" * 100)
        for participant in participants:
            print("-"*100)
            print("Participant {}'s info".format(participant[0]))
            if participant[1]:
                print("He/She has hearing impairment")
            if participant[2]:
                print("He/She has vision impairment")
            if participant[3]:
                print("He/She is an introvert")
            if participant[4]:
                print("He/She has bad internet")
            print("He/She is using {}".format(participant[5]))
            print("He/She has budget {}".format(participant[6]))
            print("He/She is in timezone {}".format(participant[7]))

    return participants


def simulate(input_meeting, input_capacity, verbose=False):
    """
    This function simulates a meeting

    :param verbose:
    :return:
    """
    # 1. load cognitive data from excel sheets
    tool_file = "data/tools.xlsx"
    requirement_file = "data/requirements.xlsx"
    tool_names, feat_names, tool_feats = process_tool_files(tool_file, debug=False)
    req_names, feat_names_copy, req_feats = process_requirement_file(requirement_file, debug=False)
    assert feat_names_copy == feat_names, "tools.xlsx and requirements.xlsx have different tool features"

    # 2. randomly creates participant
    # Sheryl: the number of participants can be either set randomly or collected from the webpage
    capacity = input_capacity
    participants = create_random_participants(capacity, verbose=verbose)

    ####################################################################################################################
    # 3. collecting_information for the meeting
    # interactions with the organizer along the way
    # (1) international meeting
    participant_timezones = [p[7] for p in participants]
    # the returned meeting_is_international is a boolean
    meeting_is_international = is_meeting_international(participant_timezones, verbose=verbose)
    # (2) vision impairment
    vision_impairments = [p[2] for p in participants]
    if np.any(vision_impairments):
        print("Make sure you always ask them whether they need any additional accomodations - they have certain tools that work best for them and it would be best to understand those constraints before the meeting.")
    # ToDo: can add more

    meeting = input_meeting
    # meeting = create_random_meeting(verbose=verbose)
    # Sheryl: instead of using the above random generator, use the function below when using data collected from the webpage
    # meeting = set_meeting(organizer_name, meeting_purpose, organizer_budget, organizer_system, organizer_timezone,
    #                       organizer_desired_tool, organizer_desired_features,
    #                       meeting_hearing_impairment, meeting_vision_impairment)
    ####################################################################################################################

    # 4. process the meeting information and participants information
    print("#"*100)
    binary_requirements, participant_systems, participant_budgets, participant_timezones = get_requirements(meeting, participants)

    # support_tools_1: (tool, price)
    support_tools_1 = process_capacity(capacity, verbose=verbose)
    # support_tools_2: (tool, lacking_systems, ac_feats)
    support_tools_2 = process_systems(tool_names, feat_names, tool_feats, participant_systems, verbose=verbose)
    # support_tools_3: (tool, price)
    support_tools_3 = process_budgets(participant_budgets, verbose=verbose)

    # combine hard constraints from above
    support_tools_13 = set(support_tools_1) & set(support_tools_3)
    # support_tools_123: (tool, price)
    support_tools_123 = []
    for tool in support_tools_13:
        if tool[0] in [t[0] for t in support_tools_2]:
            support_tools_123.append(tool)

    # 3.2 binary
    GRF = get_required_features(req_names, feat_names, req_feats, binary_requirements, verbose=verbose)
    required_features = GRF[0]
    requirements_to_feature_mapping = GRF[1]
    # add organizer's desired features

    # Tool FEATURES that are required 
    input_features = required_features + meeting[6]
    # print("INPUT FEATURES", input_features)
    final_recommendations = recommend_tools(tool_names, feat_names, tool_feats, input_features, support_tools_123, verbose=verbose)
    goal = final_recommendations[0]
    tool_feature_mapping = final_recommendations[1]
    print("MAPPING TOOLS TO FEATURES ", tool_feature_mapping)
    # goal - recommended tools/final output
    # binary_requirements - requirements gathered from organizer and participant data 
    # requirements_to_feature_mapping - mapping specific requirements that are accommodated by features in virtual tools 
    return goal, requirements_to_feature_mapping, input_features, tool_feature_mapping

if __name__ == "__main__":
    simulate(verbose=True)

