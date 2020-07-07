#!bin/python
from flask import Flask, request, render_template
from model import RegForm, testForm
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=b'\xd6\x04\xbdj\xfe\xed$c\x1e@\xad\x0f\x13,@G')
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def user_role():
    form = testForm(request.form)
    if request.method == 'POST' and form.validate_on_submit():
        req = request.form
        user_Selection = req["user_role"]
        print(user_Selection)

        return 'Thanks for letting us know! We will now direct you to the ' + user_Selection + ' page'
    return render_template('landingPage_custom.html', form=form)

if __name__ == '__main__':
    app.run()
