from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import Annotated,List,TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.types import Send
import operator
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

class Task(BaseModel):
    id:int
    title:str
    brief:str=Field(...,description="what to cover")


class Plan(BaseModel):
    blog_title:str
    tasks:List[Task]

class State(TypedDict):
    topic:str
    plan:Plan
    #reducer:result from workers get concate autometically
    sections:Annotated[List[str],operator.add]
    final:str

model=ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=5,
)

def orchestrator(state:State)->dict:
    plan=model.with_structured_output(Plan).invoke([
        SystemMessage(content=("create a blog plan with 5-7 section on the following topic")),
        HumanMessage(content=(f"topic={state['topic']}")),
    ])
    return {'plan':plan}

def fanout(state:State):
    return [Send("worker",{'task':task,"topic":state['topic'],"plan":state['plan']})
            for task in state["plan"].tasks]

def worker(payload:dict)->dict:
    task=payload["task"]
    topic=payload["topic"]
    plan=payload["plan"]

    blog=Plan.blog_title

    section_md=model.invoke([
        SystemMessage(content=("write one clean markdown section")),
        HumanMessage(content=(
            f"blog_title:{blog}\n"
            f"topic:{topic}\n"
            f"section:{task.title}\n"
            f"brief:{task.brief}\n\n"
            "return the section only in markdown"
        ))
    ]).content.strip()
    return {'section':[section_md]}

def reducer(state:State)->dict:
    title=state['plan'].blog_title
    body="\n\n".join(state["sections"]).strip()
    final_md=f"#{title}\n\n{body}\n"
    filepath=title.lower().replace(" ","-")+".md"
    output_path=Path(filepath)
    output_path.write_text(final_md,encoding="utf-8")
    return {"final":final_md}


graph=StateGraph(State)

graph.add_node("orchestrator",orchestrator)
graph.add_node("worker",worker)
graph.add_node("reducer",reducer)

graph.add_edge(START,"orchestrator")
graph.add_conditional_edges("orchestrator",fanout,["worker"])
graph.add_edge("worker","reducer")
graph.add_edge("reducer",END)

app=graph.compile()

res=app.invoke({'topic':"write a blog on self attention"})
print(res)







