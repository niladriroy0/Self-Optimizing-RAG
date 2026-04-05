import streamlit as st
import sqlite3
import pandas as pd
import json

st.title("Self Optimizing RAG Dashboard")

def load_data():
    try:
        # Use a timeout and WAL mode so reads from Streamlit don't block API writes.
        conn = sqlite3.connect("experiments.db", timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        df = pd.read_sql_query("SELECT * FROM experiments", conn)
        conn.close()
        
        # Parse the JSON config string to get chunk size and top k for grouping if needed, or keep as string
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.write("No experiments yet.")
else:
    st.subheader("Experiments")
    st.dataframe(df)

    st.subheader("Average Scores")

    st.write("Answer Relevance:", df["answer_relevance"].mean())
    st.write("Faithfulness:", df["faithfulness"].mean())

    st.subheader("Config Performance")
    
    # We can group by the config string since it is a JSON dump representation 
    st.bar_chart(df.groupby("config")["answer_relevance"].mean())