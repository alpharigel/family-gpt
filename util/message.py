
import streamlit as st

# create css sytle for the message
def message_style():
    st.markdown(
        """
        <style>
        .user-box, .ai-box {
            display: inline-block;
            padding: 10px;
            margin: 10px;
            border-radius: 10px;
            border: 1px solid transparent;
            margin-bottom: 20px;
            max-width: 80%;
            }
        .user-box {
            background-color: #464775;
            float: right;
            text-align: right;
            margin-left: 80px;
            }
        .ai-box {
            background-color: #22272E;
            float: left;
            text-align: left;
            margin-right: 80px; 
            }
        .ai-container {
            width: 100%;
            overflow-wrap: break-word;
            display: flex;
            justify-content: flex-start
            }
        .user-container {
            width: 100%;
            overflow-wrap: break-word;
            display: flex;
            justify-content: flex-end
            }
        </style>
        """, 
        unsafe_allow_html=True,
    )

# a function to display a message similar to Teams or iMessage
def message(content, is_user:bool = False, key:str = None):
    if is_user:
        st.markdown(
            f"""
            <div class="user-container">
                <div class="user-box">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ai-container">
                <div class="ai-box">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )