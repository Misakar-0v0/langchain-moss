from abc import ABC, abstractmethod
from typing import Union, List, Optional
from langchain_core.tools import BaseTool, tool
from ghostos_moss import MossRuntime, PyContext
from ghostos_container import Provider
from langchain_moss.facade import compile_moss_runtime

MOSS_INTRODUCTION = """
You are equipped with the MOSS (Model-oriented Operating System Simulator).
Which provides you a way to control your body / tools / thoughts through Python code.

basic usage: 
1. you will get the python code context that MOSS provide to you below. 
2. you can generate code with `moss` tool, then the MOSS will execute them for you.
3. if you print anything in your generated code, the output will be shown in further messages.

The python context `{modulename}` that MOSS provides to you are below:

```python
{source_code}
```

interfaces of some imported attrs are:
```python
{imported_attrs_prompt}
```

{magic_prompt_info}

{injection_info}

Notices:
* the imported functions are only shown with signature, the source code is omitted.
* the properties on moss instance, will keep existence. 
* You can bind variables of type int/float/bool/str/list/dict/BaseModel to moss instance if you need them for next turn.

You are able to call the `{name}` tool, generate code to fulfill your will.
the python code you generated, must include a `run` function, follow the pattern:

```python
def run(moss: Moss):
    \"""
    :param moss: instance of the class `Moss`, the properties on it will be injected with runtime implementations.
    :return: None
    \"""
```

Then the MOSS system will add your code to the python module provided to you, 
and execute the `run` function. 

Notices: 
* Your code will **APPEND** to the code of `{modulename}` then execute, so **DO NOT REPEAT THE DEFINED CODE IN THE MODULE**.
* if the python code context can not fulfill your will, do not use the `{name}` tool.
* you can reply as usual without calling the tool `{name}`. use it only when you know what you're doing.
* in your code generation, comments is not required, comment only when necessary.
"""

MOSS_FUNCTION_DESC = ("Useful to execute code in the python context that MOSS provide to you."
                      "The code must include a `run` function.")


class MossAction(ABC):
    """
    todo:
    """

    name: str = "moss"
    description: str = (
        "Useful to execute code in the python context that MOSS provide to you."
        "The code must include a `run` function."
    )
    return_direct: bool = False

    @abstractmethod
    def get_runtime(self) -> MossRuntime:
        pass

    def as_tool(self) -> BaseTool:
        return tool(
            name_or_callable=self.name,
            description=self.description,
            return_direct=self.return_direct
        )(self.__call__)

    def get_instruction(self) -> str:
        runtime = self.get_runtime()
        prompter = runtime.prompter()
        source_code = prompter.get_source_code()
        imported_attrs_prompt = prompter.get_imported_attrs_prompt()
        magic_prompt = prompter.get_magic_prompt()
        magic_prompt_info = ""
        if magic_prompt:
            magic_prompt_info = f"more information about the module:\n```text\n{magic_prompt}\n```\n"

        injections = runtime.moss_injections()
        injection_info = self.reflect_injections_info(injections)

        content = MOSS_INTRODUCTION.format(
            name=self.name,
            modulename=runtime.module().__name__,
            source_code=source_code,
            imported_attrs_prompt=imported_attrs_prompt,
            magic_prompt_info=magic_prompt_info,
            injection_info=injection_info
        )
        return content

    @abstractmethod
    def reflect_injections_info(self, injections: dict) -> str:
        pass

    @abstractmethod
    def save_pycontext(self, pycontext: PyContext) -> None:
        pass

    @abstractmethod
    def get_pycontext(self) -> Union[PyContext, None]:
        pass

    @abstractmethod
    def wrap_error(self, error: Union[str, Exception]) -> str:
        pass

    @abstractmethod
    def wrap_std_output(self, std_output: str) -> str:
        pass

    def __call__(self, code: str) -> str:
        """
        todo:
        """
        code = self.strip_code(code)
        if not code:
            return self.wrap_error("the moss code is empty")

        runtime = self.get_runtime()
        with runtime:
            # if code is not exists, inform the llm

            error = runtime.lint_exec_code(code)
            if error:
                return self.wrap_error(f"the moss code has syntax errors:\n{error}")

            moss = runtime.moss()
            try:
                # run the codes.
                result = runtime.execute(target="run", code=code, args=[moss])

                # check operator result
                pycontext = result.pycontext
                # rebind pycontext to session
                self.save_pycontext(pycontext)

                # handle std output
                std_output = result.std_output
                return self.wrap_std_output(std_output)

            except Exception as e:
                return self.wrap_error(e)

    @classmethod
    def strip_code(cls, code: str) -> str:
        if code.startswith("```python"):
            code = code[len("```python"):]
        if code.startswith("```"):
            code = code[len("```"):]
        if code.endswith("```"):
            code = code[:-len("```")]
        return code.strip()


class DefaultMossAction(MossAction):

    def __init__(
            self,
            modulename: str,
            providers: Optional[List[Provider]] = None,
    ):
        self.modulename = modulename
        self.providers = providers

    def get_runtime(self) -> MossRuntime:
        return compile_moss_runtime(
            self.modulename,
            providers=self.providers
        )

    def reflect_injections_info(self, injections: dict) -> str:
        return ""

    def save_pycontext(self, pycontext: PyContext) -> None:
        return None

    def get_pycontext(self) -> Union[PyContext, None]:
        return None

    def wrap_error(self, error: Union[str, Exception]) -> str:
        return f"Error during executing {self.name} code: {error}"

    def wrap_std_output(self, std_output: str) -> str:
        return f"{self.name} output:\n```text\n{std_output}\n```"
