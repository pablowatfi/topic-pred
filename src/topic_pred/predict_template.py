from typing import List, Optional
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from topic_pred.utils.preprocess_methods import encode_texts, preproc_content_texts
from topic_pred.utils.predict_method import predict_topics
from topic_pred.defaults import (
    DEFAULT_TOP_K,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_ARTIFACT_FILENAME,
    DEFAULT_THRESHOLD,
)
from pydantic import BaseModel

# This is the schema for the prediction request.
# Note you do not need to use all the fields listed.
# For example if your predictor only needs data related to the content
# then you can ignore the topic fields.
class TopicPredictionRequest(BaseModel):
    content_title: Optional[str] = None
    content_description: Optional[str] = None
    content_kind: Optional[str] = None
    content_text: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None
    topic_category: Optional[str] = None


class TopicPredictor:
    def __init__(
        self,
        artifact_path: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        """Topic predictor that loads precomputed artifacts and exposes a predict method.

        Defaults are read from `topic_pred.defaults` so they are centralized.
        """
        # project root and artifacts dir
        self.ROOT = Path.cwd()
        self.ARTIFACTS_DIR = self.ROOT / DEFAULT_ARTIFACTS_DIR

        # determine artifact file to load
        if artifact_path:
            artifact_file = Path(artifact_path)
        else:
            artifact_file = self.ARTIFACTS_DIR / DEFAULT_ARTIFACT_FILENAME

        if not artifact_file.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_file}")

        self.artifacts = joblib.load(artifact_file)

        # settings
        self.top_k = top_k
        self.min_score = DEFAULT_THRESHOLD

    def predict(self, request: TopicPredictionRequest) -> List[str]:
        """Takes in the request and can use all or some subset of input parameters to
        predict the topics associated with a given piece of content.

        Args:
            request (TopicPredictionRequest): See class TopicPredictionRequest

        Returns:
            List[str]: Should be list of topic ids.
        """

        text = preproc_content_texts(request.content_title, request.content_description)

        content_embeddings = encode_texts(text)

        # Predict using batch function
        predicted_topics_list = predict_topics(
            content_embeddings,
            self.artifacts['topic_embeddings'],
            self.artifacts['topic_ids_list'],
            min_score=self.min_score,
            top_k=self.top_k,
        )

        return predicted_topics_list
