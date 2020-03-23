import argparse
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

def main():
    parser = argparse.ArgumentParser(
        description="""
---------------------------------------------------------------
Description:
Classify a single str entry or a JSON formated data with labels
    1. soft: for soft skills like 'team player'
    2. tech: for tech skills like'python experience'
    3. none: for all other type of sentences
---------------------------------------------------------------""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True)
    parser.add_argument('-s','--str', type=str,metavar='', dest='text',
                        help='may be used to classify a single Text in prompt',
                        action='store')
    parser.add_argument('-j','--json', action='store_true',
                        help='state that a JSON file will be classified',
                        dest='fromFile')
    parser.add_argument('-p','--path', dest='path',type=str, metavar='',
                        help='path to JSON file (may contain labes)\n' +
                        "default='tech_soft_none.json'", 
                        action='store', default='tech_soft_none.json')                        

    args = parser.parse_args()
    fromFile = args.fromFile
    text = args.text
    path = args.path

    #Load trained model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")

    #Compile loaded model
    loaded_model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    #loaded_model.summary()

    #Load words used as features (BoW)
    with open('model/word_features', 'rb') as f:
        word_features = pickle.load(f)

    if (text):
        print(find_features_text(text,word_features,loaded_model)[0])

    #if (fromFile):




#Predict from entry 'doc'
def find_features_text(text,word_features,model):
    words = text
    features = []
    for w in word_features:
        features.append(1 if (w in words) else 0)
    newData = np.asarray(features).reshape(1,-1)

    predLabel = np.zeros(3)
    predRaw = np.argmax(model.predict([newData]))
    predIndex = predRaw
    predLabel[predIndex] = 1
    labels = ['none', 'soft', 'tech']
    predLabel = predLabel.reshape(1,-1)
    predLabelStr = labels[np.argmax(predLabel)]

    return predLabelStr, predLabel, predRaw

if __name__ == '__main__':
    main()