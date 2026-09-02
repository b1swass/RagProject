from pydantic import BaseModel,Field

class BlogState(BaseModel):
    user_input:str=""
    topic:str=""
    audience:str='general audience'

    #researcher output
    research:str=""
    research_feedback:str=""




    #writer output
    draft:str=''
    draft_feedback:str=""


    #editor output
    final_blog:str=""

    #metadata
    revision_count:int=0