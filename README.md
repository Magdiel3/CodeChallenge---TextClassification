# CodeChallenge---TextClassification
This code challenge is meant to evaluate my problem-solving skills in areas of text preprocessing, data manipulation and ML modeling in Python.

Text classifier: will classify the specified input, that may be a simplesentence or a whole paragraph (in german), and will output the predictedlabel as follows:
1. soft: for soft skills like 'team player'
2. tech: for tech skills like'python experience'
3. none: for all other type of sentences

textClassifier.ipynb    ->      Jupyter notebook with all the mandatory tasks.
    Tasks:
1. Read the file and prepare the data for text classification (take decisions on how to clean the data,what to do with punctuation signs or special characters, etc.)
            Used Spacy to filter out not iconic words (Stop words), punctuation and other symbols.
            Used Pandas to work easily with the data extracted from JSON file ("/model/tech_soft_none.json").
            NOTE: 'none' data is almost as double as 'tech' data and, therefore, it's implied greater misclassification from the last data type.
    2. Train a classification model that takes a sentence and predicts the class it belongs to. You are free to use any type of model or libraries for this task.
            BoW implementation with a 5000 most common words as features (using Spacy).
            Generated feature vectors with contained features for each entry (not qty, just presence).
            Label Encoding assigning numbers instead of class name (could have been omited)
            Implementation of Multilinear NBayes Model performed decently.
            Implementation of Logistic Regression with 'newton.cg' solver performed decently.
            Implementation of 1-hidden Layer Neural Network:

                    _________________________________________________________________
                    Layer (type)                 Output Shape              Param #   
                    =================================================================
                    dense_5 (Dense)              (None, 120)               600120    
                    _________________________________________________________________
                    dense_6 (Dense)              (None, 3)                 363       
                    =================================================================
                    Total params: 600,483
                    Trainable params: 600,483
                    Non-trainable params: 0
                    _________________________________________________________________        

            Training with batch size of 500 and 15 epochs.
        3. Provide evaluation metrics about the trained model, eventually visualize the results.
            Multilinear NBayes and Logistic Regression:
                Accuracy and confusion matrix displayed.
            Neural Networ:
                Displayed Confusion Matrix, grpahed Loss and Accuracy for Train and Test for all epochs.
        4. Explain your results: Is it a good model, what would you do to improve the model. 
            For the first two models, the acquired accuracy was decent, around 93% similar to the NN. The main difference was presented on the precission from each class in diverse iterations, with the selected hyperparameters defined after certain trials. NN presented overall better precission for 'tech' labels wich sometimes was assimilated by Naive Bayes, and NN also had greater 'none' precission which was soemtimes assimilated from Logistic Regression. The best of both worlds, as a singer once said.
            Improvements:   Optimize the model or define it purely with tensor flow for faster results, but current execution times are acceptable, I think.
                            Reduce features with more conditions as checking the function of the world in the sentence and reduce frequent words fromm 5000 to a lower number.
                            Test with less neurons (to reduce paramaters) to generalize even better, but right now is quite acceptable.
            NOTE: I got scared at the beggining as the models trained fairly well with no major tunings and performance wasn't compromised with BoW selection and many things that made me think I was doing something really bad. Mainly because of the NN had amazing performance since epoch 0.

classifyText.py         ->      Python file with cool promt functionality
    Description:
        The code initializes loading and setting up the model and features vector.
        The goal is that, in prompt, the user may input an extract from a Motivation Letter (or related) and it will label it, Also, when instructed, the user can copy to clipboard this text and the code will take it from there to perform the classification. If the response is not recognized, the code will tell it, as well. Just for recreational purposes.