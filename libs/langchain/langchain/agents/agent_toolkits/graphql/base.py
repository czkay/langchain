"""GraphQL agent."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.graphql.prompt import (
    GRAPHQL_PREFIX,
    GRAPHQL_SUFFIX,
)
from langchain.agents.agent_toolkits.graphql.toolkit import GraphQLToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel


def create_graphql_agent(
    llm: BaseLanguageModel,
    toolkit: GraphQLToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = GRAPHQL_PREFIX,
    suffix: str = GRAPHQL_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct an GraphQL agent from an LLM and tools.

    *Security Note*: When creating an OpenAPI agent, check the permissions
        and capabilities of the underlying toolkit.

        For example, if the default implementation of GraphQLToolkit
        uses the RequestsToolkit which contains tools to make arbitrary
        network requests against any URL (e.g., GET, POST, PATCH, PUT, DELETE),

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.
    """
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
