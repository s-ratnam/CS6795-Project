from wtforms import SubmitField, BooleanField, TextField, StringField, DecimalField, PasswordField, RadioField, IntegerField, SelectField, validators, ValidationError
from flask_wtf import Form
from wtforms.validators import Required, DataRequired

class RegForm(Form):
    name_first = StringField('First Name', [validators.DataRequired()])
    name_last = StringField('Last Name', [validators.DataRequired()])
    gender = RadioField('Gender', choices = [('Male', 'Male'), ('Female','Female')])
    email = StringField('Email Address', [validators.DataRequired(), validators.Email(), validators.Length(min=6, max=35)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Submit')

class testForm(Form):
    user_role = RadioField('User_Role', choices = [('Organizer', 'Organizer'), ('Participant','Participant')])
    submit = SubmitField('Submit')

class VA_impairments(Form):
    V_impaired = BooleanField("Visual Impairments")
    A_impaired = BooleanField("Auditory Impairments")
    submit = SubmitField('Submit')

class meetingInfo(Form):
    time_zone = SelectField('Time Zone', choices= [('US/Alaska', 'US/Alaska'), ('US/Aleutian', 'US/Aleutian'), ('US/Arizona', 'US/Arizona'), ('US/Central', 'US/Central'), ('US/East-Indiana', 'US/East-Indiana'), ('US/Eastern', 'US/Eastern'), ('US/Hawaii', 'US/Hawaii'), ('US/Mountain','US/Mountain'), ('US/Pacific', 'US/Pacific'), ('Singapore', 'Singapore')])
    
    meeting_capacity = IntegerField('Meeting Capacity (enter a number)', [validators.required()])
    meeting_purpose = SelectField('Meeting Purpose', choices = [('Presentation', 'Presentation'), ('Lecture', 'Lecture'), ('Chat', 'Chat')])
    meeting_duration = SelectField('Meeting Duration', choices = [('option1', '0 - 60 minutes'), ('option 2', '1-3 hours'), ('option 3', '3+ hours')])
    budget = DecimalField('Enter your meeting budget (US-dollar value)')
    organizer_system = SelectField('Operating System', choices = [('Mac', 'Mac'), ('Windows', 'Windows'), ('Linux', 'Linux')])
    desired_tool = SelectField('Desired Tool', choices = [('Microsoft Teams', 'Microsoft Teams'), ('Google Meet', 'Google Meet'), ('Google Meet G suite Essential', 'Google Meet G suite Essential'), ('Google Meet G suite Enterprise Essential', 'Google Meet G suite Enterprise Essential'), 
                                                            ('Google Hangouts', 'Google Hangouts'), ('Skype', 'Skype'), ('Zoom (paid)', 'Zoom (paid)'),('Zoom', 'Zoom'), ('Cisco WebEx', 'Cisco WebEx'), ('BlueJeans', 'BlueJeans'), ('Slack (paid)', 'Slack (paid)'), ('Slack', 'Slack'),
                                                            ('WhatsApp', 'WhatsApp'), ('Facetime','Facetime'), ('HouseParty', 'HouseParty')])
    # Features checklist 
    closed_captioning = BooleanField("Closed Captioning")
    screen_sharing = BooleanField("Screen Sharing")
    mute_all = BooleanField("Mute All")
    video_off = BooleanField("Video Off")
    join_browser = BooleanField("Join from Browser")
    adjustable_layout = BooleanField("Adjustable Layout")
    join_phone = BooleanField("Dial In With Phone")
    raise_hand = BooleanField("Raise Hand")
    chat_messaging = BooleanField("Chat Messaging")
    meeting_recording = BooleanField("Meeting Recording")
    polling = BooleanField("Polling/Surveys")
    virtual_background = BooleanField("Virtual Background Integration")
    screen_reader = BooleanField("Screen Reader Compatible")
    avatar = BooleanField("3D Memoji Avatar")
    live_photo = BooleanField("Live Photo")
    in_chat_games = BooleanField("In-chat games ")
   
    submit = SubmitField('Submit')

