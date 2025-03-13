from flask import Flask, render_template, request
app = Flask(__name__)

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://roshansmahesh:XRttMQjNccE2eXQv@cluster0.utjiq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true&ssl_cert_reqs=CERT_NONE"


# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.get_default_database("Tree_Failure")


@app.route('/')
def index():
  return render_template('homePage.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
  if (request.method == 'GET'):
    return render_template('contact.html')
  else:
    document = {}
    document.update(request.form)
    db.Contacts.insert_one(document)
    return render_template('contact.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
  return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'GET':
    return render_template('login.html')
  else:
    username = request.form['username']
    password = request.form['password']

@app.route('/register', methods=['GET', 'POST'])
def register():
  if request.method == 'GET':
    return render_template('register.html')
  else:
    username = request.form['username']
    password = request.form['password']

if __name__ == '__main__':
  app.run(debug=True)


