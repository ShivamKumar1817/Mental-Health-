import joblib

def main():
    print("Loading mental health classification model...")
    try:
        pipeline = joblib.load('mental_health_model_large.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have run 'train_model.py' to generate 'mental_health_model.pkl'.")
        return

    print("Model loaded successfully!")
    print("\n--- Mental Health Predictor ---")
    print("Type a statement to get a prediction, or type 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter your statement: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            print("Please enter a valid statement.")
            continue
            
        prediction = pipeline.predict([user_input])
        print(f"-> Predicted Status: {prediction[0]}")

if __name__ == "__main__":
    main()
