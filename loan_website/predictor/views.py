from django.shortcuts import render
import joblib
import os
import numpy as np
import tflite_runtime.interpreter as tflite # Naya halka engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ab .keras ki jagah .tflite load kar rahe hain
model_path = os.path.join(BASE_DIR, 'predictor', 'ml_files', 'loan_model.tflite')
scaler_path = os.path.join(BASE_DIR, 'predictor', 'ml_files', 'scaler.pkl')

scaler = joblib.load(scaler_path)

# TFLite Interpreter setup
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def home(request):
    result = None
    status = None 
    
    if request.method == 'POST':
        income = request.POST.get('ApplicantIncome')
        loan = request.POST.get('LoanAmount')
        
        if not income or not loan:
            result = "⚠️ Please enter Income and Loan Amount first!"
            status = "error"
        else:
            try:
                features = [
                    float(request.POST.get('Gender')),
                    float(request.POST.get('Married')),
                    float(request.POST.get('Dependents')),
                    float(request.POST.get('Education')),
                    float(request.POST.get('Self_Employed', 0)),
                    float(income),
                    float(request.POST.get('CoapplicantIncome', 0)),
                    float(loan),
                    float(request.POST.get('Loan_Amount_Term', 360)),
                    float(request.POST.get('Credit_History')),
                    float(request.POST.get('Property_Area')),
                ]
                
                final_features = scaler.transform([features])
                input_data = np.array(final_features, dtype=np.float32) # TFLite ko float32 chahiye
                
                # Prediction with TFLite
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                if prediction[0][0] > 0.5:
                    result = "🥳 Congratulations! Your Loan is APPROVED."
                    status = "success"
                else:
                    result = "😔 Sorry, Your Loan is REJECTED."
                    status = "fail"
            except Exception as e:
                result = f"Error: {e}"
                status = "error"

    return render(request, 'index.html', {'result': result, 'status': status})