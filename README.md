# Suggestions Microservice

The patient, in a general case, books an OPD appointment, if he doesn't know the nature of the disease he's suffering from. This leads to some redundancy in the cycle, as the OPD doctor then finally directs the patient to the specialist of the disease. We eliminate this redundancy through our filter systems microservice. This microservice has a predict disease endpoint, which takes in the symptoms input from the user, and predicts the department of the specialist which the patient should consult to. 

The various endpoints contained in this Microservice are :

- `/filter_hospitals` : This endpoint takes in the symptoms from the user. We use a pretrained Gradient Boosing Classifier model(with an accuracy of 99.3% on the test set) to predict the department of the specialist from the symptoms
