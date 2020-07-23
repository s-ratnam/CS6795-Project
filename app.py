#!bin/python
from flask import Flask, request, render_template, redirect, url_for
from model import RegForm, testForm, VA_impairments, meetingInfo
from flask_bootstrap import Bootstrap
import cognitive_model
import urllib

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=b'\xd6\x04\xbdj\xfe\xed$c\x1e@\xad\x0f\x13,@G')
app.config["MONGO_URI"] = "mongodb://localhost:161.35.8.231:8080/organizer"
# mongo = PyMongo(app)
Bootstrap(app)
global meeting_global_list
meeting_global_list = []
meeting_global_list.append("Alfred Rivera")
global visual_bool
visual_bool = False
global auditory_bool
auditory_bool = False
global overall_capacity
overall_capacity = 0 

@app.route('/', methods=['GET','POST'])
def landing_page():
    if request.method =='POST':
        return redirect(url_for('user_role'))
    return render_template('layouts/page1.html')

@app.route('/user_role', methods=['GET', 'POST'])
def user_role():
    if request.method == 'POST':
        user = request.form.get('role')
        if user == "organizer":
            return redirect(url_for('page_2'))
        if user == "participant":
            return redirect(url_for('results_no'))
        # org_id = mongo.db.organizer.insert({'role': user_Selection})
    return render_template('layouts/landingPage_custom.html')

@app.route('/page_2', methods=['GET', 'POST'])
def page_2():
    if request.method == 'POST':
        VA = request.form.getlist('impairment')
        global visual_bool
        global auditory_bool
        if 'visual-impairment' in VA:
            visual_bool = True
        if 'auditory impairment' in VA:
            auditory_bool = True
        
        return redirect(url_for('page_3'))
    return render_template('layouts/page2.html')

@app.route('/page_3', methods = ['GET', 'POST'])
def page_3():
    
    form = meetingInfo(request.form)
    if request.method == 'GET':
        form.budget.data = 0.0
        form.capacity = 0
        select_MC = 0
    if request.method == 'POST':
        select_TZ = request.form.get('timezonelist')
        select_MC = request.form.get('capacity')
        select_MD = request.form.get('meetingduration')
        select_OS = request.form.get('operatingsystem')
        features = []
        global overall_capacity
        overall_capacity = int(select_MC)
        print("THIS IS THE CAPACITY: ", overall_capacity)
        if (form.closed_captioning.data):
            features.append('Closed Captioning')
        if (form.screen_sharing.data):
            features.append('Screen Sharing')
        if (form.mute_all.data):
            features.append('Mute All')
        if (form.video_off.data):
            features.append('Video Off')
        if (form.join_browser.data):
            features.append('Join from Browser')
        if (form.adjustable_layout.data):
            features.append('Adjustable Layout')
        if (form.join_phone.data):
            features.append('Dial In With Phone')
        if (form.raise_hand.data):
            features.append('Raise Hand')
        if (form.chat_messaging.data):
            features.append('Chat Messaging')
        if (form.meeting_recording.data):
            features.append('Meeting Recording')
        if (form.polling.data):
            features.append('Polling/Surveys')
        if (form.virtual_background.data):
            features.append('Virtual Background Integration')
        if (form.screen_reader.data):
            features.append('Screen Reader Compatible')
        
        # MAINTAIN THIS ORDER OF APPENDING, JUST 
        global visual_bool
        global auditory_bool
        global meeting_global_list
        meeting_global_list.append(form.meeting_purpose.data)
        meeting_global_list.append(float(form.budget.data))
        meeting_global_list.append(select_OS)
        meeting_global_list.append(select_TZ)
        meeting_global_list.append(form.desired_tool.data)
        meeting_global_list.append(features)
        meeting_global_list.append(visual_bool)
        meeting_global_list.append(auditory_bool)
        return redirect(url_for('page_4'))
    return render_template('layouts/page3.html', form = form)

@app.route('/page_4', methods=['GET', 'POST'])
def page_4():
    if request.method == 'POST':
        user_control = request.form.get('role')
        if user_control == "Yes":
            return redirect(url_for('results'))
        if user_control =="No":
            return redirect(url_for('results_no'))
        else:
            return redirect(url_for('results_no'))

    return render_template('layouts/page4.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    global overall_capacity
    global meeting_global_list
    print("THIS IS WHERE WE AT: ", meeting_global_list)
    meeting_input = cognitive_model.set_meeting(meeting_global_list[0], meeting_global_list[1], meeting_global_list[2], meeting_global_list[3], meeting_global_list[4], meeting_global_list[5], meeting_global_list[6], meeting_global_list[7], meeting_global_list[8])
    results = cognitive_model.simulate(meeting_input, overall_capacity, verbose=True)
    result_tools = results[0]
    # overall meeting requirements based on the data collected from participants and the organizer 
    result_reqs_to_features = results[1]
    result_features = results[2]
    result_tf_mapping = results[3]
    return render_template('layouts/results.html', TFMapping = result_tf_mapping, items = result_tools, RFmapping = result_reqs_to_features, features = result_features)

@app.route('/results_no', methods=['GET', 'POST'])
def results_no():
    return render_template('layouts/results_no.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
