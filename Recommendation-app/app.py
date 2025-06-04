import streamlit as st
from src.data.load_data import movies , ratings
from src.models.recommender import recommender

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

# Streamlit UI
st.title('🎬 Hybrid Movie Recommendation System')

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Select Options")
    
    # User selection
    user_id = st.selectbox(
        "Select User ID",
        options=ratings['userId'].unique(),
        index=0
    )
    
    # Movie selection
    movie_title = st.selectbox(
        "Select a Movie You Like",
        options=movies['title'],
        index=0
    )
    
    # Number of recommendations
    num_recs = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Recommendation type
    rec_type = st.radio(
        "Recommendation Type",
        options=["Hybrid", "Content-Based", "Collaborative Filtering"],
        index=0
    )
    
    # Submit button
    submit = st.button("Get Recommendations")

with col2:
    st.subheader("Recommendations")
    
    if submit:
        with st.spinner('Generating recommendations...'):
            recommendations = recommender.recommend(
                user_id=user_id,
                title=movie_title,
                top_n=num_recs
            )
            
            if rec_type == "Hybrid":
                st.write("### Hybrid Recommendations (Combined Approach)")
                st.dataframe(recommendations['hybrid'])
                
                # Display as cards
                cols = st.columns(2)
                for idx, row in enumerate(recommendations['hybrid'].itertuples()):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; margin: 10px 0; background-color: #f0f2f6;">
                            <h4>{row.title}</h4>
                            <p>Score: {row.score}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            elif rec_type == "Content-Based":
                st.write("### Content-Based Recommendations")
                st.dataframe(recommendations['content'])
                
            else:
                st.write("### Collaborative Filtering Recommendations")
                st.dataframe(recommendations['collaborative'])
    
    else:
        st.info("Select options and click 'Get Recommendations' to see suggestions.")
