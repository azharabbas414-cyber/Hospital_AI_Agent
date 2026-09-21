
import os
import uuid
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# OPTIONAL STREAMLIT SECRET
# ============================================================

if not os.getenv("GROK_API_KEY"):

    try:

        if "GROK_API_KEY" in st.secrets:

            os.environ["GROK_API_KEY"] = (
                st.secrets["GROK_API_KEY"]
            )

    except Exception:
        pass


# ============================================================
# BACKEND
# ============================================================

from app.hospital_rag_backend import (
    hospital_ai_answer,
    load_session,
    save_session,
    clear_session
)


# ============================================================
# SESSION ID
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        "session_"
        + uuid.uuid4().hex[:10]
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = load_session(
        st.session_state.session_id
    )


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Hospital AI Assistant")

st.caption(
    "Ask questions about the hospital knowledge base."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Hospital AI")

    st.write(
        "Advanced RAG Assistant"
    )

    st.divider()

    st.write(
        f"Session ID: "
        f"`{st.session_state.session_id}`"
    )

    st.divider()

    if st.button(
        "🆕 New Chat",
        use_container_width=True
    ):

        st.session_state.session_id = (
            "session_"
            + uuid.uuid4().hex[:10]
        )

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        clear_session(
            st.session_state.session_id
        )

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.info(
        "Demo system: answers are based on "
        "the hospital knowledge base."
    )


# ============================================================
# SHOW CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"**{source['chunk_id']}**"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a hospital question..."
)


if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # AI response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching hospital knowledge..."
        ):

            try:

                result = hospital_ai_answer(
                    question=question,
                    history=st.session_state.messages
                )

                answer = result["answer"]

                st.markdown(answer)

                with st.expander(
                    "📚 Sources used"
                ):

                    for source in result["sources"]:

                        st.write(
                            f"**{source['chunk_id']}**"
                        )

                        metadata = source.get(
                            "metadata",
                            {}
                        )

                        if metadata:

                            st.caption(
                                f"Source: "
                                f"{metadata.get('source', 'N/A')} | "
                                f"Department: "
                                f"{metadata.get('department', 'N/A')}"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"]
                    }
                )

                save_session(
                    st.session_state.session_id,
                    st.session_state.messages
                )

            except Exception as e:

                error_message = (
                    "Sorry, I could not process "
                    "the question.\n\n"
                    f"Error: `{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )
