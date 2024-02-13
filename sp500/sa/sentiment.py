import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

def run():
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'
    doc = nlp(text)
    doc._.blob.polarity                            # Polarity: -0.125
    doc._.blob.subjectivity                        # Subjectivity: 0.9
    doc._.blob.sentiment_assessments.assessments   # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
    print(doc._.blob.polarity)

if __name__ == "__main__":
    run()