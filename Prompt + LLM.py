from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

chain.invoke({"foo": "bears"})

AIMessage(content="Why don't bears wear shoes?\n\nBecause they have bear feet!", additional_kwargs={}, example=False)


#PromptTemplate + LLM + OutputParser

from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()
