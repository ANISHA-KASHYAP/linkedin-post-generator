import os
import uuid
import streamlit as st
import ollama
import chromadb

EXAMPLES = """
Example 1:
I failed my first 10 interviews.

Not because I lacked skills.
Because I couldn't explain them.

Here is what changed:
- I started writing down my projects in simple words
- I practiced explaining them to a friend
- I stopped memorizing and started understanding

Interview 11? I got the offer.

Skills get you shortlisted. Clear communication gets you hired.

What is one thing that helped you in your job search?

#CareerGrowth #Interviews #LearningInPublic

Example 2:
Most beginners learn coding the wrong way.

They watch 50 tutorials.
They build zero projects.

Here is a better way:
1. Pick a small idea
2. Build it badly
3. Fix one bug at a time

You learn more in one messy project than in ten perfect tutorials.

Start small. Start today.

Are you a tutorial watcher or a project builder?

#Coding #Beginners #BuildInPublic
"""

SYSTEM_PROMPT = "You are an expert LinkedIn content writer."

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(name="my_posts")


def get_groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def ask_llama(prompt, temperature=0.9):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    key = get_groq_key()

    if key:  # online: Groq
        from groq import Groq, NotFoundError

        client = Groq(api_key=key)
        last_error = None
        for model_name in GROQ_MODELS:
            try:
                res = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
                return res.choices[0].message.content
            except NotFoundError as e:
                last_error = e  # ye model nahi mila, agla try karo
        raise last_error

    # laptop: Ollama
    res = ollama.chat(
        model="llama3.2:3b",
        messages=messages,
        options={"temperature": temperature},
    )
    return res["message"]["content"]


def show_post(text, key):
    st.code(text, language=None, wrap_lines=True)
    st.download_button(
        "Download as .txt",
        data=text,
        file_name="linkedin_post.txt",
        mime="text/plain",
        key=key,
    )


collection = get_collection()

st.title("LinkedIn Post Generator")

tab1, tab2, tab3 = st.tabs(["Generate new post", "Improve my draft", "My Style"])

# ---------- TAB 3: My Style ----------
with tab3:
    st.write("Paste your past LinkedIn posts. Separate each post with a line containing only `---`")
    raw = st.text_area("Your past posts", height=250, key="style_input")

    if st.button("Save my posts"):
        posts_list = [p.strip() for p in raw.split("---") if p.strip()]
        if posts_list:
            collection.add(
                documents=posts_list,
                ids=[str(uuid.uuid4()) for _ in posts_list],
            )
            st.success(f"Saved {len(posts_list)} posts!")
        else:
            st.warning("Please paste at least one post.")

    st.info(f"Posts saved so far: {collection.count()}")

    if st.button("Delete all my posts"):
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        st.success("All posts deleted.")
        st.rerun()

# ---------- TAB 1: Generate ----------
with tab1:
    topic = st.text_input("What's the post about?")
    tone = st.selectbox("Tone", ["Professional", "Casual", "Storytelling", "Inspirational"])
    length = st.selectbox("Length", ["Short", "Medium", "Long"])
    audience = st.text_input("Target audience (e.g. students, recruiters, founders)")
    use_emoji = st.checkbox("Use emojis", value=True)
    use_style = st.checkbox("Write in my style (uses saved posts)", value=True)
    num_variations = st.slider("Number of variations", 1, 3, 3)

    if st.button("Generate") and topic:
        emoji_rule = "Use a few relevant emojis." if use_emoji else "Do not use any emojis."

        # RAG: retrieve similar posts
        if use_style and collection.count() > 0:
            found = collection.query(
                query_texts=[topic],
                n_results=min(2, collection.count()),
            )
            style_examples = "\n\n---\n\n".join(found["documents"][0])
            st.caption(f"Using {len(found['documents'][0])} of your saved posts as style reference.")
        else:
            style_examples = EXAMPLES
            if use_style:
                st.caption("No saved posts yet, using default examples. Add posts in the My Style tab.")

        prompt = f"""Write a {length.lower()} LinkedIn post about: {topic}.
Target audience: {audience}.
Tone: {tone}. {emoji_rule}
Start with a strong hook, use short paragraphs,
end with a call to action and 3-5 relevant hashtags.

Here are examples of the style and format I want.
Copy the STYLE only. Do NOT copy the words or the topic.
{style_examples}

Now write the new post about: {topic}"""

        posts = []
        for i in range(num_variations):
            with st.spinner(f"Writing variation {i + 1}..."):
                posts.append(ask_llama(prompt, temperature=0.9))
        st.session_state["posts"] = posts

    for i, post in enumerate(st.session_state.get("posts", [])):
        st.subheader(f"Variation {i + 1}")
        show_post(post, key=f"dl_gen_{i}")
        st.divider()

# ---------- TAB 2: Improve ----------
with tab2:
    draft = st.text_area("Paste your draft here", height=200)
    goal = st.selectbox(
        "What should I improve?",
        [
            "Make the hook stronger",
            "Make it shorter and crisper",
            "Fix grammar and flow",
            "Make it more engaging",
        ],
    )

    if st.button("Improve") and draft:
        improve_prompt = f"""Here is a LinkedIn post draft:

{draft}

Improve it. Goal: {goal}.
Rules:
- Keep the original meaning and facts. Do not invent new facts or numbers.
- Use short paragraphs and simple words.
- Keep or add 3-5 relevant hashtags at the end.
- Return ONLY the improved post, nothing else.

Here are examples of the style I like. Copy the STYLE only:
{EXAMPLES}"""

        with st.spinner("Improving your draft..."):
            st.session_state["improved"] = ask_llama(improve_prompt, temperature=0.6)

    if "improved" in st.session_state:
        st.subheader("Improved post")
        show_post(st.session_state["improved"], key="dl_improved")
