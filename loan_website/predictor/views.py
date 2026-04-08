from django.shortcuts import render
import joblib
from tensorflow.keras.models import load_model
import os

# Model aur Scaler ka path set kar rahe hain
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'predictor', 'ml_files', 'loan_model.keras')
scaler_path = os.path.join(BASE_DIR, 'predictor', 'ml_files', 'scaler.pkl')

# Model aur Scaler load karna
model = load_model(model_path)
scaler = joblib.load(scaler_path)

def home(request):
    result = None
    status = None # Naya variable status track karne ke liye
    
    if request.method == 'POST':
        # Pehle check karo ki saare fields bhare hain ya nahi
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
                prediction = model.predict(final_features)
                
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