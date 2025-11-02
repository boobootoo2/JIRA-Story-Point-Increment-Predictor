**JIRA Story Point Increment Predictor**

This repo contains a public data used to train a JIRA Story Point Increment Predictor Machine Learning Model. This model uses JIRA summary and description text fields converted into embedding using hugging face public model "sentence-transformers/all-MiniLM-L6-v2". Using DataBricks AutoML this data set was analyzed and an XGBoostRegressor Model was generated that makes story point increment predictions

**Why**
For any JIRA project facing time constraints sizing JIRA issues this solution will allow a generalized prediction strategy using a complexity scale of 8 increments. However the project is mapping story points to hours is up to that team but this will give a way to assign complexity using a positional index on that scale.

**Integration**
Hosting this model on an API will allow a smooth integration from a client service sending description and summary that will be embedded and then the model returns a positional increment for the story points. The API returns a response with the increment and the client side can then map those increments to their story point scale.