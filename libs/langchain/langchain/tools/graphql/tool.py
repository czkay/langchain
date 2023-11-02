import json
from typing import Any, Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.graphql import GraphQLAPIWrapper


class BaseGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""

    graphql_wrapper: GraphQLAPIWrapper

    name: str = "query_graphql"
    description: str = """\
    Input to this tool is a detailed and correct GraphQL query, output is a result from the API.
    If the query is not correct, an error message will be returned.
    If an error is returned with 'Bad request' in it, rewrite the query and try again.
    If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.
    {example}
    {schema}\
    """  # noqa: E501

    description_example = "Example Input: query {{ allUsers {{ id, name, email }} }}"
    description_schema = "Use the following schema for your queries: \n    {schema}\n"
    auto_fetch_schema: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.graphql_wrapper = kwargs["graphql_wrapper"]
        self.auto_fetch_schema = self.graphql_wrapper.auto_fetch_schema
        self.add_schema_to_description()

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_schema_to_description(self) -> None:
        """If we decided to load the schema. Schema is included it in the prompting to the model"""
        if self.auto_fetch_schema:
            raw_schema = self.graphql_wrapper.fetch_schema()
            raw_schema = raw_schema.replace("{", "{{")
            raw_schema = raw_schema.replace("}", "}}")
            raw_schema = raw_schema.replace("\n", "\n    ")
            schema_prompt = self.description_schema.format(schema=raw_schema)
            self.description = self.description.format(
                schema=schema_prompt, example=self.description_example
            )
        else:
            self.description = self.description.format(
                example=self.description_example, schema=""
            )

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self.graphql_wrapper.run(tool_input)
            return json.dumps(result, indent=2)
        except Exception as e:
            return "Bad request: " + str(e)