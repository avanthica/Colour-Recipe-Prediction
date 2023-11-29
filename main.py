from flask import Flask, render_template, request,jsonify, url_for, redirect, send_from_directory
import pickle
import numpy as np
from clrrecpred import *
import pymongo
import os
from werkzeug.utils import secure_filename
from colorspacious import cspace_convert
from PIL import Image
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from bson import ObjectId
import json

model = pickle.load(open('clrrecpred.pkl','rb'))

database = pymongo.MongoClient('mongodb+srv://avanthimai23:Avanthica23@cluster0.b2axy70.mongodb.net/')
usersdb = database['customers']
users = usersdb.cust #external users customers for order
intusers = usersdb.intuser #internal users for prediction
orders = usersdb.order
feedbacks = usersdb.feedback

def convert_to_LAB(image_path):
    # Open the uploaded image using Pillow (PIL)
    image = Image.open(image_path)
    
    # Convert the image to an sRGBColor object
    image_rgb = sRGBColor(image.getpixel((0, 0))[0] / 255, image.getpixel((0, 0))[1] / 255, image.getpixel((0, 0))[2] / 255)
    
    # Convert sRGB to LAB color space using colormath
    lab_color = convert_color(image_rgb, LabColor)
    
    L = np.round(lab_color.lab_l,2)
    a = np.round(lab_color.lab_a,2)
    b = np.round(lab_color.lab_b,2)

    return L,a,b

def hex_to_rgb_to_lab(hex_code):
    # Remove the '#' character if it's present
    hex_code = hex_code.lstrip('#')

    # Convert the hex code to an integer
    hex_int = int(hex_code, 16)

    # Extract the RGB values using bit masking and shifting
    red = (hex_int >> 16) & 255
    green = (hex_int >> 8) & 255
    blue = hex_int & 255
    
    rgb = red,green,blue
    l, a, b= cspace_convert(rgb, start="sRGB255", end="CIELab")
    L = np.round(l,2)
    a = np.round(a,2)
    b = np.round(b,2)
    return L,a,b

def predict_dyes(L,a,b,chroma,hue,saturation,R,G,B,number_of_threads,total_thickness,substrate,abs_coeff):
    arr=np.array([[L,a,b,chroma,hue,saturation,R,G,B,number_of_threads,total_thickness,substrate,abs_coeff]],dtype=object)
    knn_output = []
    knn_output = model.predict(arr)
    
    knn_output_list = knn_output.tolist()
    
    code_1 = knn_output_list[0][0]
    code_2 = knn_output_list[0][2]
    code_3 = knn_output_list[0][4]
    
    dye_codes = []
    concentrations = []
    
    code_1_de, code_2_de, code_3_de = decode_output(code_1,code_2,code_3)
    
    dye_codes.append(code_1_de)
    dye_codes.append(code_2_de)
    dye_codes.append(code_3_de)
    
    concentrations.append(np.round(knn_output_list[0][1],2))
    concentrations.append(np.round(knn_output_list[0][3],2))
    concentrations.append(np.round(knn_output_list[0][5],2))
    

    return dye_codes,concentrations
    
    
# uploadfolder = "static"
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'jpg', 'png', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'colorrecipeprediction'

uploadfolder = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'jpg', 'png', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = uploadfolder

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feeds', methods=['POST'])
def feeds():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
    
        feed_user = {}
        feed_user['name'] = name
        feed_user['email'] = email
        feed_user['message'] = message
    
        feedbacks.insert_one(feed_user)
    
        return render_template('contact.html',data = "Feedback Submitted Successfully Thank you")

    

@app.route('/codetopredict')
def codetopredict():
    return render_template('code.html')

@app.route('/inputs',methods=['POST'])
def inputs():
    code = request.form['codecred']
    if code == 'colorrecipe':
        order_ids = [str(order['_id']) for order in orders.find({}, {'_id': 1})] 
        return render_template('input.html',order_ids=order_ids)
    else:
        return render_template('code.html',data ="Enter correct credentials")


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    return render_template("signup.html")

@app.route('/register',methods = ['GET','POST'])
def register():
    name = request.form['username']
    email = request.form['email']
    password = request.form['password']
    conpass = request.form['confirmpass']
    userss = request.form['usertype']
    
    if password != conpass:
        return render_template("signup.html",error = 'Password and confirm password does not match')
    elif name =='' or email == '' or password == '' or conpass =='' or userss == '':
        return render_template("signup.html",error ='Enter all fields')
    elif users.find_one({"email":email}) != None or intusers.find_one({"email" : email}) != None:
        return render_template("signup.html",data = 'Email already exists')
    else:
        if userss == 'External':
            reg_user ={}
            reg_user['name'] = name
            reg_user['email'] = email
            reg_user['password'] = password
            users.insert_one(reg_user)
        else:
            reg_user = {}
            reg_user['name'] = name
            reg_user['email'] = email
            reg_user['password'] = password
            intusers.insert_one(reg_user)
        return render_template("signin.html",data = "Registered Successfully. Now Login to order")
        
    
@app.route('/signin', methods = ['GET', 'POST'])
def signin():
    return render_template("signin.html")

@app.route('/login', methods = ['GET', 'POST'])
def login():
    email = request.form['email']
    password = request.form['password']
        
    ext_user_data = users.find_one({'email':email})
    int_user_data = intusers.find_one({'email':email})
    
    if ext_user_data:
        if ext_user_data['password'] == password:
            ext_name = ext_user_data['name']
            ext_email = ext_user_data['email']
            return render_template("ordersss.html",ext_name = ext_name,ext_email=ext_email)
        else:
            return render_template("signin.html",data= "Enter correct password")
    elif int_user_data:
        if int_user_data['password'] == password:
            return render_template("code.html")
        else:
            return render_template("signin.html",data= "Enter correct password")
    else:    
        return render_template("signup.html",data="Email doesn't exists. Create a new account")
                               


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/gallery')
def gallery():
    hex = []
    hex = df['hex_code'].unique()
    return render_template('shadegallery.html',data = hex)

           
@app.route('/ordered',methods=['POST'])
def ordered():
    name = request.form['username']
    email = request.form['email']
    supdate = request.form['duedate']
    phnum = request.form['phnum']
    address = request.form['address'] 
    quantity = request.form['quantity']
    hex = request.form['hexcode']
    if 'file' in request.files:
        uploaded_file = request.files['file']  
    
    if hex != '':
        hex = request.form['hexcode']
        L,a,b = hex_to_rgb_to_lab(hex)
        
    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        
        L,a,b = convert_to_LAB(file_path)
             
    substrate = request.form['substrate']
    countply = request.form['nothr']
    thickness = request.form['thickness'] 
    
    cust_ord = {}
    cust_ord['name'] = name
    cust_ord['email'] = email
    cust_ord['supply date'] = supdate
    cust_ord['phone number'] = phnum
    cust_ord['address'] = address
    cust_ord['L'] = L
    cust_ord['a']  = a
    cust_ord['b'] = b
    cust_ord['substrate'] = substrate
    cust_ord['countply'] = countply
    cust_ord['thickness'] = thickness
    cust_ord['quantity'] = quantity
    id = orders.insert_one(cust_ord).inserted_id
    
    lab = (L,a,b)
    R,G,B = create_rgb(lab)

    return render_template('invoice.html',data = [name,email,supdate,phnum,address,R,G,B,substrate,countply,thickness,quantity,id])

@app.route('/get_order/<order_id>')
def get_order(order_id):
    order = orders.find_one({'_id': ObjectId(order_id)})
    order['_id'] = str(order['_id'])
    return jsonify(order)

@app.route('/input')
def input():
    order_ids = [str(order['_id']) for order in orders.find({}, {'_id': 1})] 
    return render_template('input.html',order_ids=order_ids)

@app.route('/dyes')
def dyes():
    return render_template("siginorsignup.html")

@app.route('/predict', methods = ['POST'])
def predict():
    L = float(request.form['lval'])
    a = float(request.form['aval'])
    b = float(request.form['bval'])
    lab = (L,a,b)
    R,G,B = create_rgb(lab)
    rgb = (R,G,B)
    chroma = create_chroma(a,b)
    hue , saturation = create_huesat(rgb)
    number_of_threads = float(request.form['thrval'])
    thickness = float(request.form['thickval'])
    total_thickness = number_of_threads * thickness
    sub = request.form['substrate']
    substrate = model_sub.transform([sub])
    abs_coeff = create_abs_coeff(total_thickness,rgb)
    
    dye_codes, concentrations = predict_dyes(L,a,b,chroma,hue,saturation,R,G,B,number_of_threads,total_thickness,substrate,abs_coeff)
    
    # code_1 = dye_codes[0]
    # code_2 = dye_codes[1]
    # code_3 = dye_codes[2]
    
    # conc_1 = concentrations[0]
    # conc_2 = concentrations[1]
    # conc_3 = concentrations[2]
    
    # dat_pred = {}
    # dat_pred['L'] = L
    # dat_pred['a'] = a
    # dat_pred['b'] = b 
    # dat_pred['Number of threads'] = number_of_threads
    # dat_pred['Total thickness'] = total_thickness
    # dat_pred['Substrate'] = substrate
    
    # predi.insert_one(dat_pred)
    
    return render_template('output.html',data = [dye_codes,concentrations])
    

if __name__ == '__main__' :
    app.run(port=3000,debug = True)