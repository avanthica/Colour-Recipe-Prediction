import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colorspacious import cspace_convert
import colorsys
import math
import pickle

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("Copy of color recipe.xlsx")
df

#Preprocessing of hex codes,dye codes

def add_hex(hex_code):
  if not hex_code.startswith("#"):
    hex_code = "#" + hex_code
  else:
    hex_code
  return hex_code


for i in range(len(df["hex_code"])):
  df.iloc[i,2] = add_hex(str(df.iloc[i,2]))
  df.iloc[i,23] = add_hex(str(df.iloc[i,23]))
  df.iloc[i,25] = add_hex(str(df.iloc[i,25]))
  df.iloc[i,27] = add_hex(str(df.iloc[i,27]))

#print(df['hex_code'],df['dye_code_1'],df['dye_code_2'],df['dye_code_3'])

#print(df['hex_code'])

#df

"""#Label Encoding"""

#model_hex = preprocessing.LabelEncoder()
model_sub = preprocessing.LabelEncoder()
model_code_1 = preprocessing.LabelEncoder()
model_code_2 = preprocessing.LabelEncoder()
model_code_3 = preprocessing.LabelEncoder()

#data['shade_name_encoded'] = model.fit_transform((data['shade_name']))
#data['hex_code_encoded'] = model_hex.fit_transform((data['hex_code']))
df['substrate_encoded'] = model_sub.fit_transform((df['substrate']))
df['dye_code_1_encoded'] = model_code_1.fit_transform((df['dye_code_1']))
df['dye_code_2_encoded'] = model_code_2.fit_transform((df['dye_code_2']))
df['dye_code_3_encoded'] = model_code_3.fit_transform((df['dye_code_3']))
#df

#df.shape

#df.isnull()

#df.nunique()

#df.info()

"""#Input and Output Variables"""

cols_x = ["L","a","b","chroma","hue","saturation","R","G","B","number_of_threads","total_thickness","substrate_encoded","abs_coeff"]
cols_y = ["dye_code_1_encoded","working_conc_1","dye_code_2_encoded","working_conc_2","dye_code_3_encoded","working_conc_3"]
x = df[cols_x]
y = df[cols_y]
#print(x)
#print(y)

#x.shape

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 42)

#mapping_hex = dict(zip(data["hex_code_encoded"], data["hex_code"]))
mapping_sub = dict(zip(df["substrate_encoded"],df["substrate"]))
mapping_code_1 = dict(zip(df["dye_code_1_encoded"],df["dye_code_1"]))
mapping_code_2 = dict(zip(df["dye_code_2_encoded"],df["dye_code_2"]))
mapping_code_3 = dict(zip(df["dye_code_3_encoded"],df["dye_code_3"]))
#print(mapping_hex)
# print(mapping_sub)
# print(mapping_code_1)
# print(mapping_code_2)
# print(mapping_code_3)

def decode_output(code_1, code_2, code_3):

  output_1 = round(code_1)

  code_1_de= mapping_code_1.get(output_1)

  output_2 = round(code_2)

  code_2_de= mapping_code_2.get(output_2)

  output_3 = round(code_3)

  code_3_de= mapping_code_3.get(output_3)

  return code_1_de, code_2_de, code_3_de

"""#Feature Engineering

#LAB to RGB
"""

# Convert LAB to RGB
def create_rgb(lab):
  r, g, b= cspace_convert(lab, start="CIELab", end="sRGB255")
  R = np.round(r,0)
  G = np.round(g,0)
  B = np.round(b,0)
  #rgb = R,G,B
  #print("RGB Values","\nR : ", R ,"\nG : ", G ,"\nB : ", B)
  return R,G,B


#lab = (29.56,	33.84,	-39.70)
#create_lab(lab)

"""#RGB to Hue and Saturation"""

def create_huesat(rgb):
  # Convert RGB to HSL
  h, l, s = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

  # Convert hue from [0, 1] to [0, 360]
  hue = round(h * 360.0)
  s = round(s * 100)

  # Print HSL values with hue in degrees
  #print("Hue : ", hue,"\nSaturation : ", s)
  return hue, s

"""#Chroma"""

def create_chroma(A,B):
  # Calculate chroma
  chroma = (A ** 2 + B ** 2) ** 0.5

  # Print chroma value
  #print("Chroma:", chroma)
  return chroma

"""#Absorption Coefficient"""

def create_abs_coeff(total_thickness,rgb):

    R, G, B = rgb
    if R == 0 or G == 0 or B == 0:
      return 0
    else:
      R_1 = R / 255
      G_1 = G / 255
      B_1 = B / 255

      R_tr = 1 - R_1

      Max_R = R_tr / R_1
      Max_G = Max_R * G_1
      Max_B = Max_R * B_1

      Transmittance = (0.2125 * R_tr) + (0.7154 * Max_G) + (0.0721 * Max_B)
      if Transmittance <= 0:
        Transmittance = -(Transmittance)

      Absorbance = 2 - (math.log10(Transmittance * 100))

      Abs_coeff = (2.303 * Absorbance) / total_thickness

      #print("Absorption Coefficient : " ,Abs_coeff)
      return Abs_coeff

#rgb = (86,94,121)
#create_abs_coeff(0.075,rgb)

"""#Dye Code Representation"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def hex_to_rgb(hex_code):
    # Convert a hex color code to RGB values.
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def show_dye_codes(dye_codes):
    # Create a new figure for displaying colors
    fig, ax = plt.subplots()

    # Set aspect ratio to equal for square boxes
    ax.set_aspect('equal')

    # Initialize x-coordinate for the first box
    x = 0.1

    # Iterate over dye codes
    for color in dye_codes:
        # Convert hex color code to RGB
        rgb_color = hex_to_rgb(color)

        # Create a colored rectangle (box) for the current color
        rectangle = Rectangle((x, 0.7), 0.2, 0.2, color=color)

        # Add the rectangle to the axis
        ax.add_patch(rectangle)

        # Update x-coordinate for the next box
        x += 0.3

    # Turn off axis labels
    ax.axis('off')

    # Show the plot
    plt.title('Required Colorants or Dyestuffs')
    plt.show()

# Example dye codes
#dye_codes = ['#9CC1F5', '#F9F2F1', '#FFEF43']

# Call the show_dye_codes function with the dye codes
#show_dye_codes(dye_codes)

"""#Output Shade"""

def hex_to_rgb(hex_code):
    # Convert a hex color code to RGB values.
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def color_rep(dye_codes, concentrations):

    color1,color2,color3 = dye_codes
    concentration1,concentration2,concentration3 = concentrations
    # Convert hex color codes to RGB.
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    rgb3 = hex_to_rgb(color3)

    # Calculate the weighted average of RGB values based on concentrations.
    new_r = int((rgb1[0] * concentration1 + rgb2[0] * concentration2 + rgb3[0] * concentration3) / (concentration1 + concentration2 + concentration3))
    new_g = int((rgb1[1] * concentration1 + rgb2[1] * concentration2 + rgb3[1] * concentration3) / (concentration1 + concentration2 + concentration3))
    new_b = int((rgb1[2] * concentration1 + rgb2[2] * concentration2 + rgb3[2] * concentration3) / (concentration1 + concentration2 + concentration3))

    # Convert the resulting RGB values back to hex.
    new_hex = "#{:02X}{:02X}{:02X}".format(new_r, new_g, new_b)
    rect = patches.Rectangle((0.3, 0.7), 0.4, 0.4, color=new_hex)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add the rectangle to the axis
    ax.add_patch(rect)

    ax.axis("off")

    # Set the title
    ax.set_title("Predicted Color")
    #print(new_hex)

    # Show the plot
    plt.show()

#hex_colors = ['#9CC1F5', '#F9F2F1', '#FFEF43']
#concentrations = [0.22, 0.42, 0.57]
#color_rep(hex_colors,concentrations)

"""#Input Shade"""

def input_lab(L,a,b):

    # Convert LAB to RGB
    lab_color = [L, a, b]
    rgb_color = cspace_convert(lab_color, start="CIELab", end="sRGB1")


    # Create a colored rectangle (box)
    rect = patches.Rectangle((0.3, 0.7), 0.4, 0.4, color=rgb_color)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add the rectangle to the axis
    ax.add_patch(rect)

    ax.axis("off")

    # Set the title
    ax.set_title("Actual Color")


    # Show the plot
    plt.show()


#hex_code = "#565E79"
#input_hex(hex_code)

# Given LAB values
#L = 80.61
#a = -13.10
#b = -21.26
#80.61	-13.10	-21.26
#input_lab(L,a,b)

"""#**KNN**"""

knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors= 2))
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
print(y_pred_knn)

rmse_knn = np.sqrt(mean_squared_error(y_test,y_pred_knn))
mae_knn = mean_absolute_error(y_test,y_pred_knn)
mape_knn = mean_absolute_percentage_error(y_test,y_pred_knn)

print("RMSE : ",rmse_knn)
print("MAE : ",mae_knn)
print("MAPE : ",mape_knn)

knn_score = ("Rsquared: %.2f"% knn.score(x_test,y_test))
print (knn_score)

r2_knn =  r2_score(y_test, y_pred_knn)
print("R2 score:",r2_knn)

# L = float(input('Enter L value : '))
# a = float(input('Enter A value : '))
# b = float(input('Enter B value : '))
# lab = (L,a,b)
# R,G,B = create_rgb(lab)
# rgb = (R,G,B)
# chroma = create_chroma(a,b)
# hue , saturation = create_huesat(rgb)
# thickness = float(input('Enter thickness : '))
# number_of_threads = float(input('Enter number of threads : '))
# total_thickness = thickness * number_of_threads
# sub = input('Enter substrate : ')
# substrate = model_sub.transform([sub])
# abs_coeff = create_abs_coeff(total_thickness,rgb)

# print("\n\n\tINPUT DETAILS")
# print("LAB values : ",lab)
# print("RGB values : ",rgb)
# print("Chroma : ",chroma)
# print("Hue : ",hue)
# print("Saturation : ",saturation)
# print("Substrate : ",sub)
# print("Number of threads : ",number_of_threads)
# print("Total Thickness : ",total_thickness)
# print("Absorption Coefficient : ", abs_coeff)

# Gb_output = []

# Gb_output = knn.predict([[float(L),
#                         float(a),
#                         float(b),
#                         float(chroma),
#                         int(hue),
#                         int(saturation),
#                           int(R),
#                           int(G),
#                           int(B),
#                           float(number_of_threads),
#                         float(total_thickness),
#                         int(substrate),
#                         float(abs_coeff)]])
# print(Gb_output)
# code_1 = Gb_output[0][0]
# code_2 = Gb_output[0][2]
# code_3 = Gb_output[0][4]
# dye_codes = []
# concentrations = []
# code_1_de, code_2_de, code_3_de = decode_output(code_1,code_2,code_3)
# dye_codes.append(code_1_de)
# dye_codes.append(code_2_de)
# dye_codes.append(code_3_de)
# concentrations.append(np.round(Gb_output[0][1],2))
# concentrations.append(np.round(Gb_output[0][3],2))
# concentrations.append(np.round(Gb_output[0][5],2))

# print("\nThe Required DYE combinations and concentations are : ")
# print("Dye Code 1 : ",code_1_de)
# print("Dye Code 1 : ",np.round(Gb_output[0][1],2))
# print("Dye Code 2 : ",code_2_de)
# print("Dye Code 2 : ",np.round(Gb_output[0][3],2))
# print("Dye Code 3 : ",code_3_de)
# print("Dye Code 3 : ",np.round(Gb_output[0][5],2))

# input_lab(L,a,b)

# color_rep(dye_codes,concentrations)

# print("Colorants Required : ",dye_codes)
# print("Colorant Concentrations : ",concentrations)
# show_dye_codes(dye_codes)

pickle.dump(knn,open('clrrecpred.pkl','wb'))