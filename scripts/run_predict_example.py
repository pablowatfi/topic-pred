# scripts/run_predict_example.py
from topic_pred.predict_template import TopicPredictor, TopicPredictionRequest

def main():
    predictor = TopicPredictor()  # or TopicPredictor(artifact_path="artifacts/topic_predictor_direct_model.pkl")
    req = TopicPredictionRequest(
        content_title="Introduction to calculus",
        content_description="A short intro to derivatives and integrals with examples."
    )
    preds = predictor.predict(req)
    print("Predicted topics:", preds)

if __name__ == "__main__":
    main()
