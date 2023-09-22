from flask import Flask , request
from sentiment_classifier import SentimentClassifier
import torch 
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from transformers import BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json

app = Flask(__name__)


noResponseDict = ['No valid sentence was provided by the user']
invalidInputResponse = app.response_class(
    response = json.dumps(noResponseDict, indent=4),
    status =200,
    mimetype = 'application/json'
)

classifier = SentimentClassifier(2)

model_path = "english-abusive-MuRIL"
model = BertModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict(text):
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded_text["input_ids"].to("cpu")
    attention_mask = encoded_text["attention_mask"].to("cpu")

    with torch.no_grad():
        probabilities = F.softmax(classifier(input_ids, attention_mask), dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    predicted_class = predicted_class.cpu().item()
    print('predicyed class', predicted_class)
    print('confidence is ', confidence)
    probabilities = probabilities.flatten().cpu().numpy().tolist()
    print('probabilities ',probabilities )
    return predicted_class , confidence , probabilities


@app.route('/sentence_abusiveness', methods= ['POST'])
def detectabusiveness():
    try:
        request_body = request.get_json()
        text = request_body['text']
        predicted_class , confidence , probabilities = predict(text=text)
        value = ''
        if predicted_class==0 and probabilities[0]>0.50 and probabilities[0]<0.52 :

            value = 'Hate_Speech'
        if predicted_class==0 and probabilities[0]>0.52 :
            value = 'Higher_Hate_Speech'
        else :
            value = 'Positive'
        
        response = app.response_class(
            response= json.dumps(value , indent=4),
            status =200,
            mimetype='application/json'
        )
        return response

    except Exception as Error:
        print(Error)


if __name__ =='__main__':
    app.run(host='0.0.0.0', threaded= True, port= 7001)
    app.run(debug = True)



