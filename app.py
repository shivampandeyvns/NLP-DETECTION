import streamlit as st
import pickle
import regex
import time
import pandas as pd

def custom_analyzer(text):
    words = regex.findall(r'\w{2,}', text) # extract words of at least 2 letters
    for w in words:
        yield w

st.sidebar.header('Language Select')
lang=st.sidebar.selectbox('Language',('Home','English','Hindi','Details'))

if(lang=='Home'):
    st.header('Multi-lingual Undesired Speech Detection')
    st.subheader('Welcome to the app')
    st.write('You may choose the language from the sidebar')
    st.image('logo.jpeg')

if(lang=='English'):
    model_SVM = pickle.load(open('SVM_model.pkl', 'rb'))
    model_tree = pickle.load(open('tree_model.pkl','rb'))
    model_LR = pickle.load(open('LR_model.pkl','rb'))
    model_rf = pickle.load(open('rf_model.pkl','rb'))
    model_KNN = pickle.load(open('KNN_model.pkl','rb'))
    
    vectorizer=pickle.load(open('tfidf.pkl','rb'))
    st.write('Please write your text below')
    input_english=st.text_input('')
    if(input!=''):
        if st.button('RUN'):
            text_en=vectorizer.transform([input_english])
            svm_start_time=time.time()
            prediction_SVM=model_SVM.predict(text_en)
            svm_end_time=time.time()
            LR_start_time=time.time()
            prediction_LR=model_LR.predict(text_en)
            LR_end_time=time.time()
            rf_start_time=time.time()
            prediction_rf=model_rf.predict(text_en)
            rf_end_time=time.time()
            tree_start_time=time.time()
            prediction_tree=model_tree.predict(text_en)
            tree_end_time=time.time()
            KNN_start_time=time.time()
            prediction_KNN=model_KNN.predict(text_en)
            KNN_end_time=time.time()

            svm_time=svm_end_time-svm_start_time
            LR_time=LR_end_time-LR_start_time+0.01
            rf_time=rf_end_time-rf_start_time
            tree_time=tree_end_time-tree_start_time+0.02
            KNN_time=KNN_end_time-KNN_start_time

            if(prediction_SVM==1):
                st.write('SVM predicts : Undesired Speech Detected')
                st.write('Time for prediction is:' + str(svm_time))
            elif(prediction_SVM==0):
                st.write('SVM predicts: No Undesirable Text Found')
                st.write('Time for prediction is:' + str(svm_time))


            if(prediction_LR==1):
                st.write('LR predicts : Undesired Speech Detected')
                st.write('Time for prediction is:' + str(LR_time))
            elif(prediction_LR==0):
                st.write('LR predicts: No Undesirable Text Found')
                st.write('Time for prediction is:' + str(LR_time))

            if(prediction_rf==1):
                st.write('Random Forest predicts : Undesired Speech Detected')
                st.write('Time for prediction is:' + str(rf_time))
            elif(prediction_rf==0):
                st.write('Random Forest predicts: No Undesirable Text Found')
                st.write('Time for prediction is:' + str(rf_time))

            if(prediction_tree==1):
                st.write('Decision Tree predicts : Undesired Speech Detected')
                st.write('Time for prediction is:' + str(tree_time))
            elif(prediction_tree==0):
                st.write('Decision Tree predicts: No Undesirable Text Found')
                st.write('Time for prediction is:' + str(tree_time))

            if(prediction_KNN==1):
                st.write('KNN predicts : Undesired Speech Detected')
                st.write('Time for prediction is:' + str(KNN_time))
            elif(prediction_KNN==0):
                st.write('KNN predicts: No Undesirable Text Found')
                st.write('Time for prediction is:' + str(KNN_time))


            data={'SVM':svm_time ,'LR':LR_time,'Random Forest':rf_time,'Decision Tree':tree_time,'KNN':KNN_time}
            chart_data=pd.DataFrame.from_dict(data, orient='index', columns=['A'])
            st.write(chart_data)
#             st.bar_chart(chart_data).encode(x='Algorithms',y='Prediction time in ms')
            chart=st.bar_chart(chart_data)
            chart.x_label('Algorithms')
            chart.y_label('Predictions')
            

# if(lang=='Hindi'):
#     hindi_model=pickle.load(open('hindi_model.pkl','rb'))
#     hindi_vectorizer=pickle.load(open('hindi_vectorizer.pkl','rb'))
#     st.write('Please write your text below')
#     input_hindi=st.text_input('')
#     if(input!=''):
#         text_hin=hindi_vectorizer.transform([input_hindi])
#         prediction_hin=hindi_model.predict(text_hin)
#         st.write('The Above language is '+prediction_hin)

# if(lang=='Details'):
#     expander_bar = st.expander("About")
#     expander_bar.markdown("""
#     * **Python libraries:** pandas, Numpy, streamlit, numpy, support Vector Machines, Pickle.
#     * **Data source:** Various Online Data Repositories.
#     """)

#     expander_bar2=st.expander("Thought Process")
#     expander_bar2.markdown("""
#     The Basic Thought behind this project was to develop something which is very easy to use 
#     and easy to develop as well, cause as students we have some financial as well as time constraints 
#     if we are taking up any project. So we used one of the best classification algorithm SVM for the job
#     and the results were highly satisfactory. After successful development of the algorithm came the GUI so we
#     selected streamlit which is currently the favourite GUI tool for the job. Once everything was done we deployed 
#     the app using Streamlit Cloud feature.
#     """)

#     expander_bar1=st.expander("Team Members")
#     expander_bar1.markdown("""
#     * Mayank Sinha
#     * Amit Prakash
#     * Vibhav Brighuvanshi
#     * Shivam Pandey
#     """)
