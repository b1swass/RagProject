from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate



def get_llm(model_name:str="openai/gpt-oss-20b",temperature:float=0.5):
    api_key=os.getenv('GROQ_API_KEY')
    llm=init_chat_model(
        model=model_name,
        model_provider="groq",
        temperature=temperature,
        api_key=api_key
    )
    return llm



####research

RESEARCH_PROMPT=ChatPromptTemplate([
    {"role":"system","content":"""
    "you are a Research Agent: Given a blog topic and target audience,produce a clear,"
    "structured research outine: Include:\n"
    "1. 5-7 key points the blog should cover\n"
    "2. Important facts,stats, or examples for each point\n"
    "3. Suggested angle or hook\n"
    "be concise. Use bullet points. Do Not write the full blog yet."    
    
    """},
    {"role":"user","content":"Topic:{topic},Audience:{audience},{revision_hints}, write the research outliine now."}
])

def researcher_agent(llm:init_chat_model,topic:str,audience:str,feedback:str='')->str:
    revision_hints=f"The human provided this feedback on your previous research ,please address it {feedback}."
    if not feedback:
        revision_hints="This is your 1st attempt"

    chain= RESEARCH_PROMPT |llm
    result=chain.invoke({
        'topic':topic,
        'audience':audience,
        'revision_hints':revision_hints
    })
    return result.content




#####################  Writer Agent
##################### Writer Agent

WRITER_PROMPT = ChatPromptTemplate([
    {
        "role": "system",
        "content": """
        You are a BLOG WRITER Agent.

        Using the research notes provided, write a complete, engaging, and
        well-structured blog post for the target audience.

        Requirements:
        1. Length: 500-800 words
        2. Start with an attention-grabbing introduction.
        3. Create a clear and engaging title.
        4. Use headings and subheadings to organize the blog.
        5. Cover all important points from the research notes.
        6. Explain technical or complex ideas in simple language when necessary.
        7. Use relevant examples, statistics, and facts from the research.
        8. Keep the writing natural, informative, and engaging.
        9. Maintain a consistent tone suitable for the target audience.
        10. End with a concise conclusion that reinforces the main takeaway.
        11. Do not mention the research notes, agents, prompts, or the writing process.
        12. Do not invent facts, statistics, or sources that are not provided
            in the research notes.
        13: Use markdown Formatting
        14
        . If revision hints are provided, apply them carefully while
            maintaining the quality and accuracy of the article.

        Output only the final blog post.
        """
    },
    {
        "role": "user",
        "content": """
        Topic: {topic}
        Target Audience: {audience}

        Research Notes:
        {research}

        Revision Hints:
        {revision_hints}

        Write the complete blog post now.
        """
    }
])
def writer_agent(llm:init_chat_model,topic:str,audience:str,research:str='',feedback:str='')->str:
    revision_hints=f"The human provided this feedback on your previous draft and asked for these changes{feedback}.Please apply these changes while writing the blog."
    if not feedback:
        revision_hints="This is your 1st attempt"

    chain= WRITER_PROMPT |llm
    print(revision_hints)
    result=chain.invoke({
        'topic':topic,
        'audience':audience,
        'research':research,
        'revision_hints':revision_hints
    })
    return result.content

### editor prompr
##################### Editor Agent

##################### Editor Agent

EDITOR_PROMPT = ChatPromptTemplate([

    {
        "role": "system",
        "content": """
        You are a BLOG EDITOR Agent.

        Your job is to turn the draft into a polished blog post that feels
        naturally written by a thoughtful human writer.

        Editing Requirements:

        1. Preserve the writer's original ideas, meaning, and personality.
        2. Make the writing sound natural, conversational, and authentic.
        3. Avoid repetitive sentence structures and predictable phrasing.
        4. Vary sentence length and paragraph length naturally.
        5. Remove unnecessary filler, generic statements, and excessive
           explanations.
        6. Avoid robotic transitions such as:
           "Furthermore", "Moreover", "In conclusion", "It is important to note",
           unless they genuinely fit the context.
        7. Avoid excessive use of bullet points. Use them only when they
           genuinely improve readability.
        8. Prefer specific and concrete explanations over generic statements.
        9. Use contractions naturally when appropriate, such as "it's",
           "don't", "you'll", and "that's".
        10. Do not make every paragraph follow the same structure.
        11. Allow natural variation in tone and rhythm while remaining
            professional and appropriate for the target audience.
        12. Avoid exaggerated marketing language and empty claims.
        13. Do not force humor, opinions, or personal experiences that are
            not present in the original draft or research.
        14. Preserve facts, statistics, and examples from the research.
        15. Never invent facts, statistics, quotes, sources, or experiences.
        16. Fix grammar, spelling, punctuation, and awkward wording.
        17. Improve the introduction so it feels specific to the topic rather
            than using a generic blog introduction.
        18. Make transitions between ideas feel natural rather than formulaic.
        19. Keep the conclusion useful and concise. Do not use generic
            conclusions such as "In today's fast-paced world..."
        20. Avoid unnecessary headings if the content reads better without them.
        21. Keep the article approximately 500-800 words.
        22. Apply the revision hints when provided.

        Most importantly:
        Do not make the writing sound perfectly uniform or mechanically
        optimized. Good human writing has variation, personality, specificity,
        and a natural rhythm.

        Do not mention the editing process, agents, prompts, research notes,
        or revision hints.

        Return only the final blog post.
        """
    },

    {
        "role": "user",
        "content": """
        Topic: {topic}
        Draft Blog:
        {draft}


        Edit the draft into a natural, engaging, publication-ready blog post.
        """
    }
])

def editor_agent(llm:init_chat_model,topic:str,draft:str)->str:
    chain= EDITOR_PROMPT |llm
    result=chain.invoke({
        'topic':topic,
        'draft':draft
    })
    return result.content
