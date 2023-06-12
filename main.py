from dotenv import load_dotenv
from pyowm.commons import exceptions
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from typing import List, Union
import re
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI


load_dotenv()


class Destination_Agent:
    weather = OpenWeatherMapAPIWrapper()
    search = SerpAPIWrapper()

    template = """Provide Weather information, tourist attractions and things to explore for a valid destination.
     You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Search: You always search for the destination provided
    Thought: you should always try to find weather information and tourist attractions
    Action: Finding weather and/or tourist attractions using the tools [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    For example,
    Question: Hello AI
    Final Answer: Please enter a valid destination
    Question: Current prime minister of India
    Final Answer: Sorry, I can only inform you about Weather condition and tourist attractions
    Begin!

    These were previous tasks you completed:



    Question: {input}
    {agent_scratchpad}"""

    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search"
        ),
        Tool(
            name="Weather",
            func=weather.run,
            description="useful for when you need to know the weather"
        )

    ]

    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format_messages(self, **kwargs) -> str:
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            kwargs["agent_scratchpad"] = thoughts
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    @classmethod
    def from_llm(cls, lang_model: BaseLanguageModel, verbose: bool = True):
        llm_chain = LLMChain(llm=lang_model, prompt=cls.prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=cls.tools
        )
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=cls.tools, verbose=True)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0.5,model_name="gpt-3.5-turbo")

    weather_agent = Destination_Agent.from_llm(llm)

    # weather_agent.run("How many states in USA")
    # weather_agent.run("How many calories in peanut butter")
    # weather_agent.run("What are the chances of rainfall in London")
    try:
        weather_agent.run("Humidity in London")
    except exceptions.NotFoundError:
        print("Do not information about the destination")

    # weather_agent.run("Hello Agent")
    # weather_agent.run("Things to do in London")