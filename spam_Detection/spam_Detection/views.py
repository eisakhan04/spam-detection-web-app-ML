from django.shortcuts import render
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt_tab')
import nltk
nltk.download('punkt')




def home(request):
    return render(request , "home.html" )


def viewsfun(request):

  

    def transform_text(text):

        # convert into lower case 
        text = text.lower()
        # tokenization 
        text = nltk.word_tokenize(text)
    
        #removing special character
        y = []
        for i in text:
           if i.isalnum():
               y.append(i)
     
        # stop words and Punctuation
        text = y[:]
        y.clear()
    
        for i in text:
           if i not in stopwords.words('english') and i not in string.punctuation:
               y.append(i)
    
       # apply stemming 
        text = y[:]
        y.clear()
         
        ps = PorterStemmer() 
        for i in text:
            y.append(ps.stem(i))
        
        return " ".join(y)
    
    inputdata = request.GET.get('inputdata')
     
    
    tfidf = pickle.load(open(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Email Or SMS Spam Classifier\vectorizer.pkl" , 'rb'))
    model = pickle.load(open(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Email Or SMS Spam Classifier\model.pkl ", "rb"))
    ps = PorterStemmer()    
    
  
    # 1. Preprocessor
    transformed_data = transform_text(inputdata)

    # 2. vectorize
    victor_data = tfidf.transform([transformed_data])
    # 3. Predict
    prediction = model.predict(victor_data)
    # 4. desplay
    if prediction[0] == 1:
       predmsg = "spam message"
    else:
        predmsg = "not spam message"

    datapred = {
         'predict': predmsg ,
     }
    
   

    return render(request , "view.html" ,  datapred)
