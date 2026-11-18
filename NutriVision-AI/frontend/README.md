# Frontend integration contract

The team frontend can upload a JPEG/PNG image using POST /predict as multipart/form-data with field name image. It receives prediction_id, suggested_food, food101_top3, open_vocab_suggestions, possible_ingredients, estimated_calories, requires_confirmation and timing metadata.

Offer an editable food-name field and confirmed-ingredient choices. Submit corrections with POST /feedback and the returned prediction_id. Corrections are stored but do not automatically retrain the models.

Do not treat candidate ingredient suggestions as detected objects or the model's softmax scores as calibrated probabilities. Show a portion-size caution with calorie estimates.

This folder is reserved for the frontend team's code.
