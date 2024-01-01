import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 
import nltk 
from collections import Counter
import re 
from nltk.corpus import stopwords
import plotly.express as px
from nltk import ngrams
nltk.download('stopwords')
from langchain.memory import ConversationBufferMemory
from langchain.llms import CerebriumAI
from langchain import PromptTemplate, LLMChain
import os 
nltk.download('punkt')




# os.environ["CEREBRIUMAI_API_KEY"] = st.secrets["API_KEY"]
# cerebriumai_api_key  = "private-b167199e8e9a21241d32"
os.environ["CEREBRIUMAI_API_KEY"] = "private-b167199e8e9a21241d32"

st.set_page_config(
    page_title= "Research Launchpad", 
    page_icon="📚", 
    layout="wide", 
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.markdown("<h1 style='text-align: center;'> AI Research Analysis </h1>", unsafe_allow_html=True)


template = """
You are an AI research chatbot, you are very helpful , very consice and exact in your responses, 
you will answer questions and quries about recent developments in AI and nlp research , you will get 
paid very well for each good response
Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = CerebriumAI(
  endpoint_url="https://run.cerebrium.ai/falcon-7b-webhook/predict",
  max_length=100
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# @st.cache_data
def main(): 

    with st.sidebar: 
        #   st.title("Inspect your scraped research data and chat with a model about it")
        choice = st.radio("Navigation", ["Analysis", "Chat", "History"])
        st.info("""Inspect your scraped research data and chat with a model about it""")
        
        
        st.markdown('My Github  https://github.com/Ganryuu star the repo if you liked the project :)') 

    
    if choice == 'Analysis' : 
        st.title('1 - Dataset Inspection')
        st.subheader('checking our scraped dataset, which contains the name of Papers and the Abstract')
        data = pd.read_csv("./trending_papers.csv")
        st.write(data)
        corpus = ''.join(data.astype(str).apply(lambda x: ''.join(x), axis=1))
        words = re.findall(r'\b\w+\b', corpus.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]




        st.title('2 - Word Cloud')
        st.subheader('Inspecting the most common words in our dataset')
        wordcloud = WordCloud(width=800,
                            height=400,
                            colormap='tab20c', 
                            #   stopwords=stopwords, 
                            ).generate(corpus)


        fig, ax = plt.subplots(figsize = (20, 10))
        ax.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(fig)

        # st.markdown("*Streamlit* is **really** ***cool***.")
        st.markdown('''
            :red[Insights and conclusion]  \n 
            The worldcloud puts into perspective the most popular and trending topics in the AI research landscape right now,
            keywords like **Model**, **Image**, **Language Model**, **Data** ect.. occupy a big region in the worldcloud considering they are almost used 
            in all new topics, and proportionately we find words like **graph**, **annotation**, **segmentation** , does not appear as frequent as other words. 
            ''')



        word_counts = Counter(filtered_words)

        # Get the top 25 most common words
        top_25_words = word_counts.most_common(25)


        # Extracting words and their counts for plotting
        words, counts = zip(*top_25_words)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 25 Word Frequencies')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.savefig('wordfreq.png')


        st.title('3 - Word Frequencies')
        st.subheader('Checking the most frequent and used words in the current batch of latest research Paper')


        st.pyplot(plt)
        st.markdown('''
            :red[Insights and conclusion]  \n
            Top 25 Frequent words also support the top conclusion that some words are much more freqent and popular in the current 
            AI research landscape, we see words like :orange[Diffusion], :orange[Generation], :orange[Language] having the most frequencies due to the popularity of these topics
            ''')

        @st.cache_data
        def clean_corpus(text):
            # Replace non-word characters (anything other than letters and numbers) with a space
            cleaned_text = re.sub(r'\W+', ' ', text)
            return cleaned_text

        @st.cache_data
        def generate_ngrams(text, n):
            words = nltk.word_tokenize(text)
            return [' '.join(grams) for grams in ngrams(words, n)]

        clean_corpus = ' '.join(filtered_words)

        bigrams = generate_ngrams(clean_corpus, 2)
        trigrams = generate_ngrams(clean_corpus, 3)

        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)

        bigram_df = pd.DataFrame(bigram_freq.most_common(12), columns=['bigram', 'count'])
        trigram_df = pd.DataFrame(trigram_freq.most_common(12), columns=['trigram', 'count'])

        fig_bigram = px.bar(bigram_df, x='bigram', y='count', title='Top 10 Bigrams')
        fig_trigram = px.bar(trigram_df, x='trigram', y='count', title='Top 10 Trigrams')


        st.title("4 - N-Gram Frequency Visualization")
        st.subheader('Checking the most common words together, most pairs with 2-Grams or Bigrams, and most paired 3 word sntences or Trigrams')

        st.plotly_chart(fig_bigram, use_container_width=True)
        st.plotly_chart(fig_trigram, use_container_width=True)


        st.markdown('''
            What N-Grams enabled us to notice is the topic groupings of each research paper, the most frequently paired words together
            present exactly the topic that such paper or work is falling under . 
            ''')
    if choice == 'Chat' : 
        
        client = OpenAI(api_key=st.secrets["openai_secret"])

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        


if __name__ == "__main__": 
    main()