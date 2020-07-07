from wtforms import SubmitField, BooleanField, StringField, PasswordField, RadioField, validators
from flask_wtf import Form

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


