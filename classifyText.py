import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import model_from_json
sys.stderr = stderr
import pickle
from os import system, name 
import pyperclip

# clean prompt
def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux 
    else: 
        _ = system('clear') 

#Message from functionality
def top_message():
    clear()
    print("Text classifier: will classify the specified input, that may be a simple" +
        "sentence or a whole paragraph (in german), and will output the predicted" +
        "label as follows:\n1. soft: for soft skills like 'team player'\n2. tech: " +
        "for tech skills like'python experience'\n3. none: for all other type of " + 
        "sentences\n\nSeting up...")

top_message()
#Load trained model
print("\tLoading model JSON file.")
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print("\tSeting up model.")
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/model.h5")
print("\tLoaded model from disk.")

#Compile loaded model
loaded_model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#loaded_model.summary()
print("\tModel compiled. ")

#Load words used as features (BoW)
with open('model/word_features', 'rb') as f:
    word_features = pickle.load(f)
    print("\tLoaded features vector from Disk.")
    print("Setup done.\n")

#Predict from entry 'doc'
def find_features(doc):
    words = doc
    features = []
    for w in word_features:
        features.append(1 if (w in words) else 0)
    newData = np.asarray(features).reshape(1,-1)

    predLabel = np.zeros(3)
    predRaw = np.argmax(loaded_model.predict([newData]))
    predIndex = predRaw
    predLabel[predIndex] = 1
    labels = ['none', 'soft', 'tech']
    predLabel = predLabel.reshape(1,-1)
    predLabelStr = labels[np.argmax(predLabel)]

    return predLabelStr, predLabel, predRaw

#Initial run
rawData = input("Enter text to classify: \n")
predLabelStr,_,_ = find_features(rawData)
print("\nYour text was classified with label:  " + predLabelStr)

while(True):
    #Default header section
    sw = input("\nContinue? (Y/N) (C)  ")
    top_message()
    print("\tLoading model JSON file.")
    print("\tSeting up model.")
    print("\tLoaded model from disk.")
    print("\tModel compiled. ")
    print("\tLoaded features vector from Disk.")
    print("Setup done.\n")

    #Case when user wants to manually enter the pfrase
    if(sw == 'Y'):
        rawData = input("Enter text to classify: \n")
        predLabelStr,_,_ = find_features(rawData)
        print("\nYour text was classified with label:  " + predLabelStr)

    #Case when user wants to finish the program        
    elif(sw == 'N'):
        print("Bye :)\n\n")
        break

    #Case when user wants to predict from something previusly copied on clipboard
    elif(sw == 'C'):
        rawData = pyperclip.paste()
        predLabelStr,_,_ = find_features(rawData)
        print("Text from clipboard: \n" + rawData + "\n\nText was classified with label: " + predLabelStr)

    #Error case    
    else:
        print("--------------COMMAND ERROR--------------\nPlease enter Y or N (or C for cplipboard)\n")